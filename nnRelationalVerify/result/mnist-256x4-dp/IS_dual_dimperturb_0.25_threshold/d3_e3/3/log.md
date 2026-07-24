## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000229875


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0129776, -0.0111535, -0.0129776, -0.0111535, -0.0011818, 0.0011818)
1: (-0.0065975, -0.0060832, -0.0065975, -0.0060832, -0.0003332, 0.0003332)
2: (-0.0101181, -0.0063237, -0.0101181, -0.0063237, -0.0024584, 0.0024584)
3: (0.0002883, 0.0007905, 0.0002883, 0.0007905, -0.0003253, 0.0003253)
4: (0.0108178, 0.0136535, 0.0108178, 0.0136535, -0.0018372, 0.0018372)
5: (0.9985118, 0.9992996, 0.9985118, 0.9992996, -0.0005104, 0.0005104)
6: (0.0065328, 0.0072479, 0.0065328, 0.0072479, -0.0004633, 0.0004633)
7: (0.0009976, 0.0036664, 0.0009976, 0.0036664, -0.0017290, 0.0017290)
8: (-0.0120464, -0.0099693, -0.0120464, -0.0099693, -0.0013457, 0.0013457)
9: (-0.0031496, -0.0029704, -0.0031496, -0.0029704, -0.0001161, 0.0001161)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.60 + 1.38 = 2.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0003030, upper bound: 0.0003032

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002807, upper bound: 0.0002724
time: 0.56 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002806, upper bound: 0.0002809
time: 0.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.27 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 5, lower bound: -0.0002807, upper bound: 0.0002724
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 5, lower bound: -0.0002806, upper bound: 0.0002809

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0128213, -0.0111223, -0.0129287, -0.0111540, -0.0009617, 0.0010375
1: -0.0065535, -0.0060745, -0.0065837, -0.0060834, -0.0002712, 0.0002925
2: -0.0097931, -0.0062588, -0.0100165, -0.0063247, -0.0020006, 0.0021582
3: 0.0003313, 0.0007990, 0.0003018, 0.0007903, -0.0002647, 0.0002856
4: 0.0107693, 0.0134107, 0.0108186, 0.0135776, -0.0016129, 0.0014951
5: 0.9984983, 0.9992321, 0.9985120, 0.9992785, -0.0004481, 0.0004154
6: 0.0065205, 0.0071866, 0.0065330, 0.0072287, -0.0004068, 0.0003771
7: 0.0009520, 0.0034378, 0.0009983, 0.0035949, -0.0015179, 0.0014071
8: -0.0118685, -0.0099338, -0.0119908, -0.0099699, -0.0010951, 0.0011814
9: -0.0031527, -0.0029858, -0.0031496, -0.0029752, -0.0001019, 0.0000945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002723, upper bound: 0.0002724
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002723, upper bound: 0.0002724
time: 0.56 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0129193, -0.0111541, -0.0129726, -0.0111535, -0.0009098, 0.0011803
1: -0.0065811, -0.0060834, -0.0065961, -0.0060833, -0.0002565, 0.0003328
2: -0.0099970, -0.0063249, -0.0101078, -0.0063238, -0.0018925, 0.0024552
3: 0.0003044, 0.0007903, 0.0002897, 0.0007904, -0.0002504, 0.0003249
4: 0.0108187, 0.0135630, 0.0108179, 0.0136458, -0.0018348, 0.0014143
5: 0.9985120, 0.9992744, 0.9985118, 0.9992974, -0.0005098, 0.0003929
6: 0.0065330, 0.0072251, 0.0065328, 0.0072459, -0.0004627, 0.0003567
7: 0.0009985, 0.0035811, 0.0009977, 0.0036591, -0.0017268, 0.0013310
8: -0.0119801, -0.0099700, -0.0120407, -0.0099694, -0.0010360, 0.0013440
9: -0.0031496, -0.0029762, -0.0031496, -0.0029709, -0.0001160, 0.0000894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002724, upper bound: 0.0002807
time: 0.60 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002724, upper bound: 0.0002809
time: 0.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.77 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 5, lower bound: -0.0002723, upper bound: 0.0002724
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 5, lower bound: -0.0002723, upper bound: 0.0002724
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 5, lower bound: -0.0002724, upper bound: 0.0002807
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 5, lower bound: -0.0002724, upper bound: 0.0002809

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0128213, -0.0111223, -0.0128213, -0.0111223, -0.0008915, 0.0008915
1: -0.0065535, -0.0060745, -0.0065535, -0.0060745, -0.0002514, 0.0002514
2: -0.0097931, -0.0062588, -0.0097931, -0.0062588, -0.0018545, 0.0018545
3: 0.0003313, 0.0007990, 0.0003313, 0.0007990, -0.0002454, 0.0002454
4: 0.0107693, 0.0134107, 0.0107693, 0.0134107, -0.0013860, 0.0013860
5: 0.9984983, 0.9992321, 0.9984983, 0.9992321, -0.0003851, 0.0003851
6: 0.0065205, 0.0071866, 0.0065205, 0.0071866, -0.0003495, 0.0003495
7: 0.0009520, 0.0034378, 0.0009520, 0.0034378, -0.0013044, 0.0013044
8: -0.0118685, -0.0099338, -0.0118685, -0.0099338, -0.0010152, 0.0010152
9: -0.0031527, -0.0029858, -0.0031527, -0.0029858, -0.0000876, 0.0000876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002640, upper bound: 0.0002591
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002643, upper bound: 0.0002643
time: 0.59 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0128213, -0.0111223, -0.0129193, -0.0111541, -0.0009617, 0.0011004
1: -0.0065535, -0.0060745, -0.0065811, -0.0060834, -0.0002711, 0.0003103
2: -0.0097931, -0.0062588, -0.0099970, -0.0063249, -0.0020005, 0.0022891
3: 0.0003313, 0.0007990, 0.0003044, 0.0007903, -0.0002647, 0.0003029
4: 0.0107693, 0.0134107, 0.0108187, 0.0135630, -0.0017108, 0.0014950
5: 0.9984983, 0.9992321, 0.9985120, 0.9992744, -0.0004753, 0.0004154
6: 0.0065205, 0.0071866, 0.0065330, 0.0072251, -0.0004314, 0.0003770
7: 0.0009520, 0.0034378, 0.0009985, 0.0035811, -0.0016100, 0.0014070
8: -0.0118685, -0.0099338, -0.0119801, -0.0099700, -0.0010951, 0.0012531
9: -0.0031527, -0.0029858, -0.0031496, -0.0029762, -0.0001081, 0.0000945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002593, upper bound: 0.0002641
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002642, upper bound: 0.0002644
time: 0.61 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0129193, -0.0111541, -0.0128213, -0.0111223, -0.0011004, 0.0009617
1: -0.0065811, -0.0060834, -0.0065535, -0.0060745, -0.0003103, 0.0002711
2: -0.0099970, -0.0063249, -0.0097931, -0.0062588, -0.0022891, 0.0020005
3: 0.0003044, 0.0007903, 0.0003313, 0.0007990, -0.0003029, 0.0002647
4: 0.0108187, 0.0135630, 0.0107693, 0.0134107, -0.0014950, 0.0017108
5: 0.9985120, 0.9992744, 0.9984983, 0.9992321, -0.0004154, 0.0004753
6: 0.0065330, 0.0072251, 0.0065205, 0.0071866, -0.0003770, 0.0004314
7: 0.0009985, 0.0035811, 0.0009520, 0.0034378, -0.0014070, 0.0016100
8: -0.0119801, -0.0099700, -0.0118685, -0.0099338, -0.0012531, 0.0010951
9: -0.0031496, -0.0029762, -0.0031527, -0.0029858, -0.0000945, 0.0001081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002641, upper bound: 0.0002688
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002642, upper bound: 0.0002727
time: 0.56 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0129193, -0.0111541, -0.0129193, -0.0111541, -0.0009092, 0.0009092
1: -0.0065811, -0.0060834, -0.0065811, -0.0060834, -0.0002563, 0.0002563
2: -0.0099970, -0.0063249, -0.0099970, -0.0063249, -0.0018914, 0.0018914
3: 0.0003044, 0.0007903, 0.0003044, 0.0007903, -0.0002503, 0.0002503
4: 0.0108187, 0.0135630, 0.0108187, 0.0135630, -0.0014135, 0.0014135
5: 0.9985120, 0.9992744, 0.9985120, 0.9992744, -0.0003927, 0.0003927
6: 0.0065330, 0.0072251, 0.0065330, 0.0072251, -0.0003565, 0.0003565
7: 0.0009985, 0.0035811, 0.0009985, 0.0035811, -0.0013303, 0.0013303
8: -0.0119801, -0.0099700, -0.0119801, -0.0099700, -0.0010354, 0.0010354
9: -0.0031496, -0.0029762, -0.0031496, -0.0029762, -0.0000893, 0.0000893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002641, upper bound: 0.0002690
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002642, upper bound: 0.0002725
time: 0.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.84 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 5, lower bound: -0.0002640, upper bound: 0.0002591
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 5, lower bound: -0.0002643, upper bound: 0.0002643
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 5, lower bound: -0.0002593, upper bound: 0.0002641
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 5, lower bound: -0.0002642, upper bound: 0.0002644
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 5, lower bound: -0.0002641, upper bound: 0.0002688
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 5, lower bound: -0.0002642, upper bound: 0.0002727
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 5, lower bound: -0.0002641, upper bound: 0.0002690
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 5, lower bound: -0.0002642, upper bound: 0.0002725

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0127428, -0.0110968, -0.0127990, -0.0111235, -0.0007919, 0.0008676
1: -0.0065313, -0.0060673, -0.0065472, -0.0060748, -0.0002233, 0.0002446
2: -0.0096297, -0.0062058, -0.0097468, -0.0062614, -0.0016472, 0.0018047
3: 0.0003530, 0.0008061, 0.0003375, 0.0007987, -0.0002180, 0.0002388
4: 0.0107297, 0.0132885, 0.0107712, 0.0133760, -0.0013487, 0.0012310
5: 0.9984873, 0.9991982, 0.9984989, 0.9992225, -0.0003747, 0.0003420
6: 0.0065105, 0.0071558, 0.0065210, 0.0071779, -0.0003401, 0.0003104
7: 0.0009147, 0.0033228, 0.0009538, 0.0034052, -0.0012693, 0.0011585
8: -0.0117790, -0.0099048, -0.0118431, -0.0099352, -0.0009017, 0.0009879
9: -0.0031552, -0.0029935, -0.0031526, -0.0029880, -0.0000852, 0.0000778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002589, upper bound: 0.0002488
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002591, upper bound: 0.0002543
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128067, -0.0111231, -0.0128213, -0.0111223, -0.0008197, 0.0008898
1: -0.0065493, -0.0060747, -0.0065535, -0.0060745, -0.0002311, 0.0002509
2: -0.0097627, -0.0062604, -0.0097931, -0.0062588, -0.0017051, 0.0018510
3: 0.0003354, 0.0007988, 0.0003313, 0.0007990, -0.0002256, 0.0002450
4: 0.0107705, 0.0133879, 0.0107693, 0.0134107, -0.0013833, 0.0012742
5: 0.9984986, 0.9992259, 0.9984983, 0.9992321, -0.0003843, 0.0003540
6: 0.0065208, 0.0071809, 0.0065205, 0.0071866, -0.0003489, 0.0003213
7: 0.0009531, 0.0034164, 0.0009520, 0.0034378, -0.0013019, 0.0011992
8: -0.0118518, -0.0099347, -0.0118685, -0.0099338, -0.0009333, 0.0010133
9: -0.0031526, -0.0029872, -0.0031527, -0.0029858, -0.0000874, 0.0000805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002592, upper bound: 0.0002518
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002593, upper bound: 0.0002593
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0127990, -0.0111235, -0.0128452, -0.0111250, -0.0009506, 0.0010225
1: -0.0065472, -0.0060748, -0.0065602, -0.0060752, -0.0002680, 0.0002883
2: -0.0097468, -0.0062614, -0.0098428, -0.0062644, -0.0019774, 0.0021271
3: 0.0003375, 0.0007987, 0.0003248, 0.0007983, -0.0002617, 0.0002815
4: 0.0107712, 0.0133760, 0.0107735, 0.0134478, -0.0015897, 0.0014778
5: 0.9984989, 0.9992225, 0.9984994, 0.9992424, -0.0004417, 0.0004106
6: 0.0065210, 0.0071779, 0.0065216, 0.0071960, -0.0004009, 0.0003727
7: 0.0009538, 0.0034052, 0.0009559, 0.0034727, -0.0014960, 0.0013908
8: -0.0118431, -0.0099352, -0.0118957, -0.0099369, -0.0010824, 0.0011644
9: -0.0031526, -0.0029880, -0.0031524, -0.0029834, -0.0001005, 0.0000934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002637, upper bound: 0.0002518
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002637, upper bound: 0.0002592
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0128213, -0.0111223, -0.0129027, -0.0111549, -0.0009600, 0.0010496
1: -0.0065535, -0.0060745, -0.0065764, -0.0060836, -0.0002707, 0.0002959
2: -0.0097931, -0.0062588, -0.0099624, -0.0063266, -0.0019970, 0.0021833
3: 0.0003313, 0.0007990, 0.0003089, 0.0007901, -0.0002643, 0.0002889
4: 0.0107693, 0.0134107, 0.0108200, 0.0135372, -0.0016317, 0.0014924
5: 0.9984983, 0.9992321, 0.9985123, 0.9992673, -0.0004533, 0.0004146
6: 0.0065205, 0.0071866, 0.0065333, 0.0072185, -0.0004115, 0.0003764
7: 0.0009520, 0.0034378, 0.0009997, 0.0035568, -0.0015356, 0.0014045
8: -0.0118685, -0.0099338, -0.0119612, -0.0099709, -0.0010931, 0.0011952
9: -0.0031527, -0.0029858, -0.0031495, -0.0029778, -0.0001031, 0.0000943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002675, upper bound: 0.0002520
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002676, upper bound: 0.0002593
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0128452, -0.0111250, -0.0127990, -0.0111235, -0.0010225, 0.0009506
1: -0.0065602, -0.0060752, -0.0065472, -0.0060748, -0.0002883, 0.0002680
2: -0.0098428, -0.0062644, -0.0097468, -0.0062614, -0.0021271, 0.0019774
3: 0.0003248, 0.0007983, 0.0003375, 0.0007987, -0.0002815, 0.0002617
4: 0.0107735, 0.0134478, 0.0107712, 0.0133760, -0.0014778, 0.0015897
5: 0.9984994, 0.9992424, 0.9984989, 0.9992225, -0.0004106, 0.0004417
6: 0.0065216, 0.0071960, 0.0065210, 0.0071779, -0.0003727, 0.0004009
7: 0.0009559, 0.0034727, 0.0009538, 0.0034052, -0.0013908, 0.0014960
8: -0.0118957, -0.0099369, -0.0118431, -0.0099352, -0.0011644, 0.0010824
9: -0.0031524, -0.0029834, -0.0031526, -0.0029880, -0.0000934, 0.0001005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002517, upper bound: 0.0002637
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002590, upper bound: 0.0002638
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0129027, -0.0111549, -0.0128213, -0.0111223, -0.0010496, 0.0009600
1: -0.0065764, -0.0060836, -0.0065535, -0.0060745, -0.0002959, 0.0002707
2: -0.0099624, -0.0063266, -0.0097931, -0.0062588, -0.0021833, 0.0019970
3: 0.0003089, 0.0007901, 0.0003313, 0.0007990, -0.0002889, 0.0002643
4: 0.0108200, 0.0135372, 0.0107693, 0.0134107, -0.0014924, 0.0016317
5: 0.9985123, 0.9992673, 0.9984983, 0.9992321, -0.0004146, 0.0004533
6: 0.0065333, 0.0072185, 0.0065205, 0.0071866, -0.0003764, 0.0004115
7: 0.0009997, 0.0035568, 0.0009520, 0.0034378, -0.0014045, 0.0015356
8: -0.0119612, -0.0099709, -0.0118685, -0.0099338, -0.0011952, 0.0010931
9: -0.0031495, -0.0029778, -0.0031527, -0.0029858, -0.0000943, 0.0001031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002520, upper bound: 0.0002675
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002593, upper bound: 0.0002677
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0128452, -0.0111250, -0.0128985, -0.0111554, -0.0008095, 0.0008880
1: -0.0065602, -0.0060752, -0.0065752, -0.0060838, -0.0002282, 0.0002504
2: -0.0098428, -0.0062644, -0.0099536, -0.0063277, -0.0016839, 0.0018472
3: 0.0003248, 0.0007983, 0.0003101, 0.0007899, -0.0002228, 0.0002445
4: 0.0107735, 0.0134478, 0.0108208, 0.0135306, -0.0013805, 0.0012584
5: 0.9984994, 0.9992424, 0.9985126, 0.9992654, -0.0003835, 0.0003496
6: 0.0065216, 0.0071960, 0.0065335, 0.0072169, -0.0003481, 0.0003174
7: 0.0009559, 0.0034727, 0.0010005, 0.0035507, -0.0012992, 0.0011843
8: -0.0118957, -0.0099369, -0.0119563, -0.0099715, -0.0009218, 0.0010112
9: -0.0031524, -0.0029834, -0.0031494, -0.0029782, -0.0000872, 0.0000795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002591, upper bound: 0.0002589
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002591, upper bound: 0.0002640
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0129027, -0.0111549, -0.0129193, -0.0111541, -0.0008384, 0.0009076
1: -0.0065764, -0.0060836, -0.0065811, -0.0060834, -0.0002364, 0.0002559
2: -0.0099624, -0.0063266, -0.0099970, -0.0063249, -0.0017440, 0.0018880
3: 0.0003089, 0.0007901, 0.0003044, 0.0007903, -0.0002308, 0.0002499
4: 0.0108200, 0.0135372, 0.0108187, 0.0135630, -0.0014110, 0.0013034
5: 0.9985123, 0.9992673, 0.9985120, 0.9992744, -0.0003920, 0.0003621
6: 0.0065333, 0.0072185, 0.0065330, 0.0072251, -0.0003558, 0.0003287
7: 0.0009997, 0.0035568, 0.0009985, 0.0035811, -0.0013279, 0.0012266
8: -0.0119612, -0.0099709, -0.0119801, -0.0099700, -0.0009547, 0.0010335
9: -0.0031495, -0.0029778, -0.0031496, -0.0029762, -0.0000892, 0.0000824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002593, upper bound: 0.0002611
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002594, upper bound: 0.0002678
time: 0.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.84 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 5, lower bound: -0.0002589, upper bound: 0.0002488
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 5, lower bound: -0.0002591, upper bound: 0.0002543
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 5, lower bound: -0.0002592, upper bound: 0.0002518
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 5, lower bound: -0.0002593, upper bound: 0.0002593
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 5, lower bound: -0.0002637, upper bound: 0.0002518
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 5, lower bound: -0.0002637, upper bound: 0.0002592
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 5, lower bound: -0.0002675, upper bound: 0.0002520
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 5, lower bound: -0.0002676, upper bound: 0.0002593
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 5, lower bound: -0.0002517, upper bound: 0.0002637
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 5, lower bound: -0.0002590, upper bound: 0.0002638
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 5, lower bound: -0.0002520, upper bound: 0.0002675
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 5, lower bound: -0.0002593, upper bound: 0.0002677
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 5, lower bound: -0.0002591, upper bound: 0.0002589
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 5, lower bound: -0.0002591, upper bound: 0.0002640
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 5, lower bound: -0.0002593, upper bound: 0.0002611
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 5, lower bound: -0.0002594, upper bound: 0.0002678

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0127018, -0.0110922, -0.0127858, -0.0111236, -0.0007469, 0.0008504
1: -0.0065198, -0.0060660, -0.0065435, -0.0060748, -0.0002106, 0.0002398
2: -0.0095445, -0.0061961, -0.0097193, -0.0062616, -0.0015536, 0.0017690
3: 0.0003642, 0.0008073, 0.0003411, 0.0007987, -0.0002056, 0.0002341
4: 0.0107225, 0.0132248, 0.0107714, 0.0133555, -0.0013221, 0.0011611
5: 0.9984852, 0.9991804, 0.9984989, 0.9992168, -0.0003673, 0.0003226
6: 0.0065087, 0.0071398, 0.0065211, 0.0071727, -0.0003334, 0.0002928
7: 0.0009079, 0.0032629, 0.0009539, 0.0033859, -0.0012442, 0.0010927
8: -0.0117324, -0.0098995, -0.0118281, -0.0099353, -0.0008504, 0.0009684
9: -0.0031557, -0.0029975, -0.0031526, -0.0029893, -0.0000835, 0.0000734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002413, upper bound: 0.0002397
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002551, upper bound: 0.0002447
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0127273, -0.0110970, -0.0127976, -0.0111236, -0.0007558, 0.0008605
1: -0.0065270, -0.0060673, -0.0065468, -0.0060748, -0.0002131, 0.0002426
2: -0.0095975, -0.0062062, -0.0097438, -0.0062614, -0.0015722, 0.0017901
3: 0.0003572, 0.0008060, 0.0003379, 0.0007987, -0.0002081, 0.0002369
4: 0.0107300, 0.0132644, 0.0107713, 0.0133738, -0.0013378, 0.0011750
5: 0.9984874, 0.9991915, 0.9984989, 0.9992218, -0.0003717, 0.0003264
6: 0.0065106, 0.0071498, 0.0065210, 0.0071773, -0.0003374, 0.0002963
7: 0.0009150, 0.0033002, 0.0009538, 0.0034030, -0.0012590, 0.0011058
8: -0.0117614, -0.0099050, -0.0118415, -0.0099352, -0.0008606, 0.0009799
9: -0.0031552, -0.0029950, -0.0031526, -0.0029881, -0.0000845, 0.0000743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002517, upper bound: 0.0002542
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002517, upper bound: 0.0002542
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0127641, -0.0111211, -0.0128083, -0.0111224, -0.0007745, 0.0008700
1: -0.0065373, -0.0060741, -0.0065498, -0.0060745, -0.0002184, 0.0002453
2: -0.0096741, -0.0062564, -0.0097660, -0.0062591, -0.0016112, 0.0018097
3: 0.0003471, 0.0007994, 0.0003349, 0.0007990, -0.0002132, 0.0002395
4: 0.0107675, 0.0133217, 0.0107695, 0.0133904, -0.0013525, 0.0012041
5: 0.9984977, 0.9992074, 0.9984984, 0.9992266, -0.0003758, 0.0003345
6: 0.0065201, 0.0071642, 0.0065206, 0.0071815, -0.0003411, 0.0003037
7: 0.0009503, 0.0033541, 0.0009522, 0.0034187, -0.0012728, 0.0011332
8: -0.0118033, -0.0099325, -0.0118537, -0.0099339, -0.0008820, 0.0009906
9: -0.0031528, -0.0029914, -0.0031527, -0.0029871, -0.0000855, 0.0000761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002541, upper bound: 0.0002517
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002541, upper bound: 0.0002520
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0127918, -0.0111233, -0.0128199, -0.0111223, -0.0007872, 0.0008827
1: -0.0065452, -0.0060747, -0.0065531, -0.0060745, -0.0002219, 0.0002489
2: -0.0097317, -0.0062608, -0.0097901, -0.0062589, -0.0016376, 0.0018361
3: 0.0003395, 0.0007988, 0.0003317, 0.0007990, -0.0002167, 0.0002430
4: 0.0107708, 0.0133648, 0.0107694, 0.0134084, -0.0013722, 0.0012238
5: 0.9984987, 0.9992194, 0.9984983, 0.9992315, -0.0003812, 0.0003400
6: 0.0065209, 0.0071751, 0.0065205, 0.0071861, -0.0003461, 0.0003086
7: 0.0009534, 0.0033946, 0.0009520, 0.0034356, -0.0012914, 0.0011517
8: -0.0118349, -0.0099349, -0.0118668, -0.0099338, -0.0008964, 0.0010051
9: -0.0031526, -0.0029887, -0.0031527, -0.0029859, -0.0000867, 0.0000773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002542, upper bound: 0.0002591
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002542, upper bound: 0.0002593
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0127571, -0.0111216, -0.0128327, -0.0111251, -0.0009038, 0.0010048
1: -0.0065354, -0.0060742, -0.0065567, -0.0060752, -0.0002548, 0.0002833
2: -0.0096595, -0.0062573, -0.0098167, -0.0062646, -0.0018801, 0.0020902
3: 0.0003490, 0.0007992, 0.0003282, 0.0007983, -0.0002488, 0.0002766
4: 0.0107682, 0.0133108, 0.0107737, 0.0134283, -0.0015621, 0.0014051
5: 0.9984979, 0.9992044, 0.9984995, 0.9992370, -0.0004340, 0.0003904
6: 0.0065202, 0.0071614, 0.0065216, 0.0071911, -0.0003939, 0.0003543
7: 0.0009509, 0.0033438, 0.0009561, 0.0034543, -0.0014701, 0.0013224
8: -0.0117953, -0.0099330, -0.0118814, -0.0099370, -0.0010292, 0.0011442
9: -0.0031528, -0.0029921, -0.0031524, -0.0029847, -0.0000987, 0.0000888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002585, upper bound: 0.0002303
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002600, upper bound: 0.0002478
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127841, -0.0111237, -0.0128437, -0.0111250, -0.0009161, 0.0010149
1: -0.0065430, -0.0060748, -0.0065598, -0.0060752, -0.0002583, 0.0002861
2: -0.0097157, -0.0062617, -0.0098396, -0.0062645, -0.0019056, 0.0021112
3: 0.0003416, 0.0007987, 0.0003252, 0.0007983, -0.0002522, 0.0002794
4: 0.0107715, 0.0133528, 0.0107735, 0.0134454, -0.0015778, 0.0014241
5: 0.9984989, 0.9992160, 0.9984994, 0.9992418, -0.0004384, 0.0003957
6: 0.0065211, 0.0071720, 0.0065216, 0.0071954, -0.0003979, 0.0003591
7: 0.0009540, 0.0033833, 0.0009560, 0.0034705, -0.0014849, 0.0013403
8: -0.0118261, -0.0099354, -0.0118940, -0.0099369, -0.0010431, 0.0011557
9: -0.0031526, -0.0029894, -0.0031524, -0.0029836, -0.0000997, 0.0000900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002586, upper bound: 0.0002591
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002586, upper bound: 0.0002591
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0127793, -0.0111204, -0.0128900, -0.0111550, -0.0009120, 0.0010343
1: -0.0065416, -0.0060739, -0.0065728, -0.0060837, -0.0002571, 0.0002916
2: -0.0097058, -0.0062549, -0.0099359, -0.0063268, -0.0018972, 0.0021516
3: 0.0003429, 0.0007996, 0.0003124, 0.0007900, -0.0002511, 0.0002847
4: 0.0107664, 0.0133454, 0.0108201, 0.0135174, -0.0016080, 0.0014179
5: 0.9984975, 0.9992140, 0.9985123, 0.9992618, -0.0004468, 0.0003939
6: 0.0065198, 0.0071702, 0.0065333, 0.0072135, -0.0004055, 0.0003576
7: 0.0009492, 0.0033763, 0.0009998, 0.0035382, -0.0015133, 0.0013344
8: -0.0118207, -0.0099317, -0.0119467, -0.0099710, -0.0010385, 0.0011778
9: -0.0031529, -0.0029899, -0.0031495, -0.0029790, -0.0001016, 0.0000896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002636, upper bound: 0.0002483
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002636, upper bound: 0.0002520
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128065, -0.0111225, -0.0129012, -0.0111549, -0.0009224, 0.0010419
1: -0.0065493, -0.0060745, -0.0065760, -0.0060836, -0.0002601, 0.0002938
2: -0.0097623, -0.0062592, -0.0099593, -0.0063266, -0.0019188, 0.0021674
3: 0.0003354, 0.0007990, 0.0003093, 0.0007901, -0.0002539, 0.0002868
4: 0.0107696, 0.0133876, 0.0108200, 0.0135348, -0.0016198, 0.0014340
5: 0.9984984, 0.9992257, 0.9985123, 0.9992666, -0.0004500, 0.0003984
6: 0.0065206, 0.0071808, 0.0065333, 0.0072180, -0.0004085, 0.0003616
7: 0.0009523, 0.0034161, 0.0009997, 0.0035546, -0.0015244, 0.0013495
8: -0.0118516, -0.0099340, -0.0119595, -0.0099709, -0.0010503, 0.0011864
9: -0.0031527, -0.0029872, -0.0031495, -0.0029779, -0.0001024, 0.0000906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002637, upper bound: 0.0002537
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002637, upper bound: 0.0002593
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0128327, -0.0111251, -0.0127571, -0.0111216, -0.0010048, 0.0009038
1: -0.0065567, -0.0060752, -0.0065354, -0.0060742, -0.0002833, 0.0002548
2: -0.0098167, -0.0062646, -0.0096595, -0.0062573, -0.0020902, 0.0018801
3: 0.0003282, 0.0007983, 0.0003490, 0.0007992, -0.0002766, 0.0002488
4: 0.0107737, 0.0134283, 0.0107682, 0.0133108, -0.0014051, 0.0015621
5: 0.9984995, 0.9992370, 0.9984979, 0.9992044, -0.0003904, 0.0004340
6: 0.0065216, 0.0071911, 0.0065202, 0.0071614, -0.0003543, 0.0003939
7: 0.0009561, 0.0034543, 0.0009509, 0.0033438, -0.0013224, 0.0014701
8: -0.0118814, -0.0099370, -0.0117953, -0.0099330, -0.0011442, 0.0010292
9: -0.0031524, -0.0029847, -0.0031528, -0.0029921, -0.0000888, 0.0000987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002303, upper bound: 0.0002585
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002477, upper bound: 0.0002600
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0128437, -0.0111250, -0.0127841, -0.0111237, -0.0010149, 0.0009161
1: -0.0065598, -0.0060752, -0.0065430, -0.0060748, -0.0002861, 0.0002583
2: -0.0098396, -0.0062645, -0.0097157, -0.0062617, -0.0021112, 0.0019056
3: 0.0003252, 0.0007983, 0.0003416, 0.0007987, -0.0002794, 0.0002522
4: 0.0107735, 0.0134454, 0.0107715, 0.0133528, -0.0014241, 0.0015778
5: 0.9984994, 0.9992418, 0.9984989, 0.9992160, -0.0003957, 0.0004384
6: 0.0065216, 0.0071954, 0.0065211, 0.0071720, -0.0003591, 0.0003979
7: 0.0009560, 0.0034705, 0.0009540, 0.0033833, -0.0013403, 0.0014849
8: -0.0118940, -0.0099369, -0.0118261, -0.0099354, -0.0011557, 0.0010431
9: -0.0031524, -0.0029836, -0.0031526, -0.0029894, -0.0000900, 0.0000997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002589, upper bound: 0.0002587
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002589, upper bound: 0.0002638
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0128900, -0.0111550, -0.0127793, -0.0111204, -0.0010343, 0.0009120
1: -0.0065728, -0.0060837, -0.0065416, -0.0060739, -0.0002916, 0.0002571
2: -0.0099359, -0.0063268, -0.0097058, -0.0062549, -0.0021516, 0.0018972
3: 0.0003124, 0.0007900, 0.0003429, 0.0007996, -0.0002847, 0.0002511
4: 0.0108201, 0.0135174, 0.0107664, 0.0133454, -0.0014179, 0.0016080
5: 0.9985123, 0.9992618, 0.9984975, 0.9992140, -0.0003939, 0.0004468
6: 0.0065333, 0.0072135, 0.0065198, 0.0071702, -0.0003576, 0.0004055
7: 0.0009998, 0.0035382, 0.0009492, 0.0033763, -0.0013344, 0.0015133
8: -0.0119467, -0.0099710, -0.0118207, -0.0099317, -0.0011778, 0.0010385
9: -0.0031495, -0.0029790, -0.0031529, -0.0029899, -0.0000896, 0.0001016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002482, upper bound: 0.0002673
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002482, upper bound: 0.0002676
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0129012, -0.0111549, -0.0128065, -0.0111225, -0.0010419, 0.0009224
1: -0.0065760, -0.0060836, -0.0065493, -0.0060745, -0.0002938, 0.0002601
2: -0.0099593, -0.0063266, -0.0097623, -0.0062592, -0.0021674, 0.0019188
3: 0.0003093, 0.0007901, 0.0003354, 0.0007990, -0.0002868, 0.0002539
4: 0.0108200, 0.0135348, 0.0107696, 0.0133876, -0.0014340, 0.0016198
5: 0.9985123, 0.9992666, 0.9984984, 0.9992257, -0.0003984, 0.0004500
6: 0.0065333, 0.0072180, 0.0065206, 0.0071808, -0.0003616, 0.0004085
7: 0.0009997, 0.0035546, 0.0009523, 0.0034161, -0.0013495, 0.0015244
8: -0.0119595, -0.0099709, -0.0118516, -0.0099340, -0.0011864, 0.0010503
9: -0.0031495, -0.0029779, -0.0031527, -0.0029872, -0.0000906, 0.0001024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002537, upper bound: 0.0002675
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002537, upper bound: 0.0002677
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0128037, -0.0111212, -0.0128858, -0.0111555, -0.0007647, 0.0008709
1: -0.0065485, -0.0060741, -0.0065716, -0.0060838, -0.0002156, 0.0002455
2: -0.0097564, -0.0062565, -0.0099272, -0.0063279, -0.0015907, 0.0018116
3: 0.0003362, 0.0007993, 0.0003136, 0.0007899, -0.0002105, 0.0002397
4: 0.0107676, 0.0133832, 0.0108210, 0.0135108, -0.0013538, 0.0011888
5: 0.9984978, 0.9992245, 0.9985126, 0.9992599, -0.0003761, 0.0003303
6: 0.0065201, 0.0071797, 0.0065336, 0.0072119, -0.0003414, 0.0002998
7: 0.0009504, 0.0034119, 0.0010006, 0.0035320, -0.0012741, 0.0011188
8: -0.0118484, -0.0099326, -0.0119419, -0.0099716, -0.0008707, 0.0009916
9: -0.0031528, -0.0029875, -0.0031494, -0.0029795, -0.0000856, 0.0000751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002408, upper bound: 0.0002537
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002553, upper bound: 0.0002552
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0128298, -0.0111252, -0.0128970, -0.0111555, -0.0007738, 0.0008810
1: -0.0065559, -0.0060753, -0.0065748, -0.0060838, -0.0002182, 0.0002484
2: -0.0098108, -0.0062648, -0.0099505, -0.0063278, -0.0016097, 0.0018327
3: 0.0003290, 0.0007983, 0.0003105, 0.0007899, -0.0002130, 0.0002425
4: 0.0107738, 0.0134239, 0.0108208, 0.0135283, -0.0013696, 0.0012030
5: 0.9984995, 0.9992357, 0.9985126, 0.9992648, -0.0003805, 0.0003342
6: 0.0065217, 0.0071900, 0.0065335, 0.0072163, -0.0003454, 0.0003034
7: 0.0009562, 0.0034502, 0.0010005, 0.0035485, -0.0012890, 0.0011322
8: -0.0118782, -0.0099371, -0.0119547, -0.0099715, -0.0008812, 0.0010032
9: -0.0031524, -0.0029849, -0.0031494, -0.0029783, -0.0000866, 0.0000760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002519, upper bound: 0.0002639
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002519, upper bound: 0.0002639
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0128615, -0.0111492, -0.0129067, -0.0111542, -0.0007932, 0.0008886
1: -0.0065648, -0.0060820, -0.0065775, -0.0060834, -0.0002236, 0.0002505
2: -0.0098766, -0.0063148, -0.0099708, -0.0063251, -0.0016500, 0.0018485
3: 0.0003203, 0.0007916, 0.0003078, 0.0007903, -0.0002184, 0.0002446
4: 0.0108112, 0.0134731, 0.0108189, 0.0135434, -0.0013815, 0.0012331
5: 0.9985099, 0.9992495, 0.9985121, 0.9992690, -0.0003838, 0.0003426
6: 0.0065311, 0.0072024, 0.0065330, 0.0072201, -0.0003484, 0.0003110
7: 0.0009914, 0.0034965, 0.0009986, 0.0035627, -0.0013001, 0.0011605
8: -0.0119142, -0.0099645, -0.0119657, -0.0099701, -0.0009032, 0.0010119
9: -0.0031501, -0.0029818, -0.0031496, -0.0029774, -0.0000873, 0.0000779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002539, upper bound: 0.0002610
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002539, upper bound: 0.0002612
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0128873, -0.0111551, -0.0129178, -0.0111541, -0.0008065, 0.0009006
1: -0.0065721, -0.0060837, -0.0065807, -0.0060834, -0.0002274, 0.0002539
2: -0.0099304, -0.0063269, -0.0099939, -0.0063249, -0.0016777, 0.0018734
3: 0.0003132, 0.0007900, 0.0003048, 0.0007903, -0.0002220, 0.0002479
4: 0.0108202, 0.0135132, 0.0108187, 0.0135607, -0.0014000, 0.0012538
5: 0.9985124, 0.9992606, 0.9985120, 0.9992738, -0.0003890, 0.0003483
6: 0.0065334, 0.0072125, 0.0065330, 0.0072245, -0.0003531, 0.0003162
7: 0.0009999, 0.0035343, 0.0009985, 0.0035789, -0.0013176, 0.0011800
8: -0.0119436, -0.0099711, -0.0119784, -0.0099700, -0.0009184, 0.0010255
9: -0.0031495, -0.0029793, -0.0031496, -0.0029763, -0.0000885, 0.0000792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002538, upper bound: 0.0002675
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002538, upper bound: 0.0002677
time: 0.60 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.03 seconds
IS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002413, upper bound: 0.0002397
IS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002551, upper bound: 0.0002447
IS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002517, upper bound: 0.0002542
IS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002517, upper bound: 0.0002542
IS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002541, upper bound: 0.0002517
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002541, upper bound: 0.0002520
IS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002542, upper bound: 0.0002591
IS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002542, upper bound: 0.0002593
IS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002585, upper bound: 0.0002303
IS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002600, upper bound: 0.0002478
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002586, upper bound: 0.0002591
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002586, upper bound: 0.0002591
IS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002636, upper bound: 0.0002483
IS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002636, upper bound: 0.0002520
IS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002637, upper bound: 0.0002537
IS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002637, upper bound: 0.0002593
IS_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002303, upper bound: 0.0002585
IS_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002477, upper bound: 0.0002600
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002589, upper bound: 0.0002587
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002589, upper bound: 0.0002638
IS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002482, upper bound: 0.0002673
IS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002482, upper bound: 0.0002676
IS_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002537, upper bound: 0.0002675
IS_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002537, upper bound: 0.0002677
IS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002408, upper bound: 0.0002537
IS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002553, upper bound: 0.0002552
IS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002519, upper bound: 0.0002639
IS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002519, upper bound: 0.0002639
IS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002539, upper bound: 0.0002610
IS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002539, upper bound: 0.0002612
IS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002538, upper bound: 0.0002675
IS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 5, lower bound: -0.0002538, upper bound: 0.0002677

## BFS IS instance: IS_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0126913, -0.0110924, -0.0127495, -0.0111159, -0.0007357, 0.0008091
1: -0.0065168, -0.0060660, -0.0065332, -0.0060726, -0.0002074, 0.0002281
2: -0.0095226, -0.0061967, -0.0096438, -0.0062454, -0.0015304, 0.0016830
3: 0.0003671, 0.0008073, 0.0003511, 0.0008008, -0.0002025, 0.0002227
4: 0.0107229, 0.0132085, 0.0107593, 0.0132990, -0.0012578, 0.0011437
5: 0.9984854, 0.9991759, 0.9984955, 0.9992011, -0.0003494, 0.0003178
6: 0.0065088, 0.0071357, 0.0065180, 0.0071585, -0.0003172, 0.0002884
7: 0.0009083, 0.0032475, 0.0009426, 0.0033327, -0.0011837, 0.0010763
8: -0.0117204, -0.0098998, -0.0117867, -0.0099265, -0.0008377, 0.0009213
9: -0.0031556, -0.0029986, -0.0031533, -0.0029928, -0.0000795, 0.0000723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002397
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002397
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0127018, -0.0110922, -0.0127771, -0.0111243, -0.0007462, 0.0008148
1: -0.0065198, -0.0060660, -0.0065410, -0.0060750, -0.0002104, 0.0002297
2: -0.0095445, -0.0061961, -0.0097012, -0.0062629, -0.0015523, 0.0016950
3: 0.0003642, 0.0008073, 0.0003435, 0.0007985, -0.0002054, 0.0002243
4: 0.0107225, 0.0132248, 0.0107723, 0.0133419, -0.0012668, 0.0011601
5: 0.9984852, 0.9991804, 0.9984991, 0.9992130, -0.0003519, 0.0003223
6: 0.0065087, 0.0071398, 0.0065213, 0.0071693, -0.0003195, 0.0002926
7: 0.0009079, 0.0032629, 0.0009548, 0.0033731, -0.0011922, 0.0010918
8: -0.0117324, -0.0098995, -0.0118182, -0.0099360, -0.0008497, 0.0009279
9: -0.0031557, -0.0029975, -0.0031525, -0.0029901, -0.0000801, 0.0000733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002514, upper bound: 0.0002448
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002514, upper bound: 0.0002448
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127273, -0.0110970, -0.0127571, -0.0111216, -0.0007777, 0.0008167
1: -0.0065270, -0.0060673, -0.0065354, -0.0060742, -0.0002193, 0.0002303
2: -0.0095975, -0.0062062, -0.0096595, -0.0062573, -0.0016178, 0.0016990
3: 0.0003572, 0.0008060, 0.0003490, 0.0007992, -0.0002141, 0.0002248
4: 0.0107300, 0.0132644, 0.0107682, 0.0133108, -0.0012697, 0.0012091
5: 0.9984874, 0.9991915, 0.9984979, 0.9992044, -0.0003528, 0.0003359
6: 0.0065106, 0.0071498, 0.0065202, 0.0071614, -0.0003202, 0.0003049
7: 0.0009150, 0.0033002, 0.0009509, 0.0033438, -0.0011949, 0.0011379
8: -0.0117614, -0.0099050, -0.0117953, -0.0099330, -0.0008856, 0.0009300
9: -0.0031552, -0.0029950, -0.0031528, -0.0029921, -0.0000802, 0.0000764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002303, upper bound: 0.0002472
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002477, upper bound: 0.0002500
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127273, -0.0110970, -0.0127841, -0.0111237, -0.0007512, 0.0008288
1: -0.0065270, -0.0060673, -0.0065430, -0.0060748, -0.0002118, 0.0002337
2: -0.0095975, -0.0062062, -0.0097157, -0.0062617, -0.0015627, 0.0017240
3: 0.0003572, 0.0008060, 0.0003416, 0.0007987, -0.0002068, 0.0002281
4: 0.0107300, 0.0132644, 0.0107715, 0.0133528, -0.0012884, 0.0011678
5: 0.9984874, 0.9991915, 0.9984989, 0.9992160, -0.0003580, 0.0003245
6: 0.0065106, 0.0071498, 0.0065211, 0.0071720, -0.0003249, 0.0002945
7: 0.0009150, 0.0033002, 0.0009540, 0.0033833, -0.0012126, 0.0010991
8: -0.0117614, -0.0099050, -0.0118261, -0.0099354, -0.0008554, 0.0009437
9: -0.0031552, -0.0029950, -0.0031526, -0.0029894, -0.0000814, 0.0000738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002303, upper bound: 0.0002480
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002477, upper bound: 0.0002501
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0127641, -0.0111211, -0.0127300, -0.0110969, -0.0008449, 0.0007721
1: -0.0065373, -0.0060741, -0.0065277, -0.0060673, -0.0002382, 0.0002177
2: -0.0096741, -0.0062564, -0.0096031, -0.0062060, -0.0017576, 0.0016061
3: 0.0003471, 0.0007994, 0.0003565, 0.0008060, -0.0002326, 0.0002125
4: 0.0107675, 0.0133217, 0.0107299, 0.0132686, -0.0012003, 0.0013135
5: 0.9984977, 0.9992074, 0.9984873, 0.9991927, -0.0003335, 0.0003649
6: 0.0065201, 0.0071642, 0.0065106, 0.0071508, -0.0003027, 0.0003312
7: 0.0009503, 0.0033541, 0.0009149, 0.0033041, -0.0011296, 0.0012361
8: -0.0118033, -0.0099325, -0.0117644, -0.0099049, -0.0009621, 0.0008792
9: -0.0031528, -0.0029914, -0.0031552, -0.0029948, -0.0000758, 0.0000830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002472, upper bound: 0.0002303
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002501, upper bound: 0.0002478
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0127641, -0.0111211, -0.0127935, -0.0111232, -0.0007729, 0.0008012
1: -0.0065373, -0.0060741, -0.0065456, -0.0060747, -0.0002179, 0.0002259
2: -0.0096741, -0.0062564, -0.0097353, -0.0062606, -0.0016078, 0.0016666
3: 0.0003471, 0.0007994, 0.0003390, 0.0007988, -0.0002128, 0.0002205
4: 0.0107675, 0.0133217, 0.0107707, 0.0133674, -0.0012455, 0.0012016
5: 0.9984977, 0.9992074, 0.9984986, 0.9992201, -0.0003460, 0.0003338
6: 0.0065201, 0.0071642, 0.0065209, 0.0071757, -0.0003141, 0.0003030
7: 0.0009503, 0.0033541, 0.0009533, 0.0033971, -0.0011721, 0.0011308
8: -0.0118033, -0.0099325, -0.0118368, -0.0099348, -0.0008801, 0.0009123
9: -0.0031528, -0.0029914, -0.0031526, -0.0029885, -0.0000787, 0.0000759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002479
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002501, upper bound: 0.0002479
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127918, -0.0111233, -0.0127413, -0.0110968, -0.0008566, 0.0007844
1: -0.0065452, -0.0060747, -0.0065309, -0.0060673, -0.0002415, 0.0002212
2: -0.0097317, -0.0062608, -0.0096266, -0.0062059, -0.0017819, 0.0016318
3: 0.0003395, 0.0007988, 0.0003534, 0.0008060, -0.0002358, 0.0002159
4: 0.0107708, 0.0133648, 0.0107297, 0.0132862, -0.0012195, 0.0013317
5: 0.9984987, 0.9992194, 0.9984873, 0.9991975, -0.0003388, 0.0003700
6: 0.0065209, 0.0071751, 0.0065106, 0.0071553, -0.0003075, 0.0003358
7: 0.0009534, 0.0033946, 0.0009147, 0.0033206, -0.0011477, 0.0012532
8: -0.0118349, -0.0099349, -0.0117773, -0.0099048, -0.0009754, 0.0008932
9: -0.0031526, -0.0029887, -0.0031552, -0.0029936, -0.0000771, 0.0000842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002488, upper bound: 0.0002590
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002488, upper bound: 0.0002591
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127918, -0.0111233, -0.0128053, -0.0111231, -0.0007856, 0.0008109
1: -0.0065452, -0.0060747, -0.0065489, -0.0060747, -0.0002215, 0.0002286
2: -0.0097317, -0.0062608, -0.0097597, -0.0062604, -0.0016343, 0.0016868
3: 0.0003395, 0.0007988, 0.0003358, 0.0007988, -0.0002163, 0.0002232
4: 0.0107708, 0.0133648, 0.0107705, 0.0133857, -0.0012606, 0.0012214
5: 0.9984987, 0.9992194, 0.9984986, 0.9992252, -0.0003502, 0.0003393
6: 0.0065209, 0.0071751, 0.0065208, 0.0071803, -0.0003179, 0.0003080
7: 0.0009534, 0.0033946, 0.0009531, 0.0034143, -0.0011864, 0.0011495
8: -0.0118349, -0.0099349, -0.0118502, -0.0099347, -0.0008946, 0.0009234
9: -0.0031526, -0.0029887, -0.0031526, -0.0029874, -0.0000797, 0.0000772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002488, upper bound: 0.0002591
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002488, upper bound: 0.0002594
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0127209, -0.0111130, -0.0128216, -0.0111254, -0.0008615, 0.0009952
1: -0.0065252, -0.0060718, -0.0065535, -0.0060753, -0.0002429, 0.0002806
2: -0.0095843, -0.0062394, -0.0097936, -0.0062652, -0.0017921, 0.0020702
3: 0.0003590, 0.0008016, 0.0003313, 0.0007982, -0.0002372, 0.0002740
4: 0.0107548, 0.0132546, 0.0107741, 0.0134110, -0.0015471, 0.0013393
5: 0.9984943, 0.9991887, 0.9984996, 0.9992322, -0.0004298, 0.0003721
6: 0.0065169, 0.0071473, 0.0065217, 0.0071867, -0.0003902, 0.0003378
7: 0.0009383, 0.0032909, 0.0009565, 0.0034381, -0.0014560, 0.0012604
8: -0.0117542, -0.0099232, -0.0118688, -0.0099373, -0.0009810, 0.0011332
9: -0.0031536, -0.0029956, -0.0031524, -0.0029858, -0.0000978, 0.0000846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002585, upper bound: 0.0002201
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002585, upper bound: 0.0002303
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0127483, -0.0111222, -0.0128327, -0.0111251, -0.0008716, 0.0010042
1: -0.0065329, -0.0060744, -0.0065567, -0.0060752, -0.0002457, 0.0002831
2: -0.0096412, -0.0062586, -0.0098167, -0.0062646, -0.0018131, 0.0020890
3: 0.0003514, 0.0007991, 0.0003282, 0.0007983, -0.0002399, 0.0002765
4: 0.0107692, 0.0132971, 0.0107737, 0.0134283, -0.0015612, 0.0013550
5: 0.9984982, 0.9992006, 0.9984995, 0.9992370, -0.0004338, 0.0003765
6: 0.0065205, 0.0071580, 0.0065216, 0.0071911, -0.0003937, 0.0003417
7: 0.0009518, 0.0033309, 0.0009561, 0.0034543, -0.0014693, 0.0012752
8: -0.0117853, -0.0099337, -0.0118814, -0.0099370, -0.0009925, 0.0011435
9: -0.0031527, -0.0029930, -0.0031524, -0.0029847, -0.0000987, 0.0000856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002599, upper bound: 0.0002446
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002599, upper bound: 0.0002477
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127841, -0.0111237, -0.0128037, -0.0111212, -0.0009390, 0.0009750
1: -0.0065430, -0.0060748, -0.0065485, -0.0060741, -0.0002647, 0.0002749
2: -0.0097157, -0.0062617, -0.0097564, -0.0062565, -0.0019533, 0.0020282
3: 0.0003416, 0.0007987, 0.0003362, 0.0007993, -0.0002585, 0.0002684
4: 0.0107715, 0.0133528, 0.0107676, 0.0133832, -0.0015157, 0.0014598
5: 0.9984989, 0.9992160, 0.9984978, 0.9992245, -0.0004211, 0.0004056
6: 0.0065211, 0.0071720, 0.0065201, 0.0071797, -0.0003822, 0.0003681
7: 0.0009540, 0.0033833, 0.0009504, 0.0034119, -0.0014265, 0.0013738
8: -0.0118261, -0.0099354, -0.0118484, -0.0099326, -0.0010692, 0.0011102
9: -0.0031526, -0.0029894, -0.0031528, -0.0029875, -0.0000958, 0.0000922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002529, upper bound: 0.0002413
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002549, upper bound: 0.0002551
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127841, -0.0111237, -0.0128298, -0.0111252, -0.0009120, 0.0009848
1: -0.0065430, -0.0060748, -0.0065559, -0.0060753, -0.0002571, 0.0002777
2: -0.0097157, -0.0062617, -0.0098108, -0.0062648, -0.0018972, 0.0020486
3: 0.0003416, 0.0007987, 0.0003290, 0.0007983, -0.0002511, 0.0002711
4: 0.0107715, 0.0133528, 0.0107738, 0.0134239, -0.0015310, 0.0014178
5: 0.9984989, 0.9992160, 0.9984995, 0.9992357, -0.0004254, 0.0003939
6: 0.0065211, 0.0071720, 0.0065217, 0.0071900, -0.0003861, 0.0003576
7: 0.0009540, 0.0033833, 0.0009562, 0.0034502, -0.0014408, 0.0013343
8: -0.0118261, -0.0099354, -0.0118782, -0.0099371, -0.0010385, 0.0011214
9: -0.0031526, -0.0029894, -0.0031524, -0.0029849, -0.0000967, 0.0000896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002529, upper bound: 0.0002418
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002549, upper bound: 0.0002553
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0127018, -0.0110922, -0.0128900, -0.0111550, -0.0008164, 0.0010772
1: -0.0065198, -0.0060660, -0.0065728, -0.0060837, -0.0002302, 0.0003037
2: -0.0095445, -0.0061961, -0.0099359, -0.0063268, -0.0016984, 0.0022407
3: 0.0003642, 0.0008073, 0.0003124, 0.0007900, -0.0002248, 0.0002965
4: 0.0107225, 0.0132248, 0.0108201, 0.0135174, -0.0016746, 0.0012693
5: 0.9984852, 0.9991804, 0.9985123, 0.9992618, -0.0004652, 0.0003526
6: 0.0065087, 0.0071398, 0.0065333, 0.0072135, -0.0004223, 0.0003201
7: 0.0009079, 0.0032629, 0.0009998, 0.0035382, -0.0015759, 0.0011945
8: -0.0117324, -0.0098995, -0.0119467, -0.0099710, -0.0009297, 0.0012266
9: -0.0031557, -0.0029975, -0.0031495, -0.0029790, -0.0001058, 0.0000802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002321, upper bound: 0.0002396
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002598, upper bound: 0.0002442
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0127641, -0.0111211, -0.0128900, -0.0111550, -0.0008558, 0.0010328
1: -0.0065373, -0.0060741, -0.0065728, -0.0060837, -0.0002413, 0.0002912
2: -0.0096741, -0.0062564, -0.0099359, -0.0063268, -0.0017803, 0.0021484
3: 0.0003471, 0.0007994, 0.0003124, 0.0007900, -0.0002356, 0.0002843
4: 0.0107675, 0.0133217, 0.0108201, 0.0135174, -0.0016056, 0.0013305
5: 0.9984977, 0.9992074, 0.9985123, 0.9992618, -0.0004461, 0.0003697
6: 0.0065201, 0.0071642, 0.0065333, 0.0072135, -0.0004049, 0.0003355
7: 0.0009503, 0.0033541, 0.0009998, 0.0035382, -0.0015110, 0.0012521
8: -0.0118033, -0.0099325, -0.0119467, -0.0099710, -0.0009745, 0.0011761
9: -0.0031528, -0.0029914, -0.0031495, -0.0029790, -0.0001015, 0.0000841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002585, upper bound: 0.0002349
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002600, upper bound: 0.0002480
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0127273, -0.0110970, -0.0129012, -0.0111549, -0.0008277, 0.0010854
1: -0.0065270, -0.0060673, -0.0065760, -0.0060836, -0.0002334, 0.0003060
2: -0.0095975, -0.0062062, -0.0099593, -0.0063266, -0.0017218, 0.0022578
3: 0.0003572, 0.0008060, 0.0003093, 0.0007901, -0.0002278, 0.0002988
4: 0.0107300, 0.0132644, 0.0108200, 0.0135348, -0.0016873, 0.0012867
5: 0.9984874, 0.9991915, 0.9985123, 0.9992666, -0.0004688, 0.0003575
6: 0.0065106, 0.0071498, 0.0065333, 0.0072180, -0.0004255, 0.0003245
7: 0.0009150, 0.0033002, 0.0009997, 0.0035546, -0.0015880, 0.0012110
8: -0.0117614, -0.0099050, -0.0119595, -0.0099709, -0.0009425, 0.0012359
9: -0.0031552, -0.0029950, -0.0031495, -0.0029779, -0.0001066, 0.0000813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002587, upper bound: 0.0002536
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002587, upper bound: 0.0002537
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0127918, -0.0111233, -0.0129012, -0.0111549, -0.0008690, 0.0010403
1: -0.0065452, -0.0060747, -0.0065760, -0.0060836, -0.0002450, 0.0002933
2: -0.0097317, -0.0062608, -0.0099593, -0.0063266, -0.0018077, 0.0021641
3: 0.0003395, 0.0007988, 0.0003093, 0.0007901, -0.0002392, 0.0002864
4: 0.0107708, 0.0133648, 0.0108200, 0.0135348, -0.0016173, 0.0013509
5: 0.9984987, 0.9992194, 0.9985123, 0.9992666, -0.0004493, 0.0003753
6: 0.0065209, 0.0071751, 0.0065333, 0.0072180, -0.0004079, 0.0003407
7: 0.0009534, 0.0033946, 0.0009997, 0.0035546, -0.0015221, 0.0012714
8: -0.0118349, -0.0099349, -0.0119595, -0.0099709, -0.0009895, 0.0011846
9: -0.0031526, -0.0029887, -0.0031495, -0.0029779, -0.0001022, 0.0000854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002587, upper bound: 0.0002592
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002587, upper bound: 0.0002593
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0128216, -0.0111254, -0.0127209, -0.0111130, -0.0009952, 0.0008615
1: -0.0065535, -0.0060753, -0.0065252, -0.0060718, -0.0002806, 0.0002429
2: -0.0097936, -0.0062652, -0.0095843, -0.0062394, -0.0020702, 0.0017921
3: 0.0003313, 0.0007982, 0.0003590, 0.0008016, -0.0002740, 0.0002372
4: 0.0107741, 0.0134110, 0.0107548, 0.0132546, -0.0013393, 0.0015471
5: 0.9984996, 0.9992322, 0.9984943, 0.9991887, -0.0003721, 0.0004298
6: 0.0065217, 0.0071867, 0.0065169, 0.0071473, -0.0003378, 0.0003902
7: 0.0009565, 0.0034381, 0.0009383, 0.0032909, -0.0012604, 0.0014560
8: -0.0118688, -0.0099373, -0.0117542, -0.0099232, -0.0011332, 0.0009810
9: -0.0031524, -0.0029858, -0.0031536, -0.0029956, -0.0000846, 0.0000978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002585
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002585
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0128327, -0.0111251, -0.0127483, -0.0111222, -0.0010042, 0.0008716
1: -0.0065567, -0.0060752, -0.0065329, -0.0060744, -0.0002831, 0.0002457
2: -0.0098167, -0.0062646, -0.0096412, -0.0062586, -0.0020890, 0.0018131
3: 0.0003282, 0.0007983, 0.0003514, 0.0007991, -0.0002765, 0.0002399
4: 0.0107737, 0.0134283, 0.0107692, 0.0132971, -0.0013550, 0.0015612
5: 0.9984995, 0.9992370, 0.9984982, 0.9992006, -0.0003765, 0.0004338
6: 0.0065216, 0.0071911, 0.0065205, 0.0071580, -0.0003417, 0.0003937
7: 0.0009561, 0.0034543, 0.0009518, 0.0033309, -0.0012752, 0.0014693
8: -0.0118814, -0.0099370, -0.0117853, -0.0099337, -0.0011435, 0.0009925
9: -0.0031524, -0.0029847, -0.0031527, -0.0029930, -0.0000856, 0.0000987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002446, upper bound: 0.0002600
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002446, upper bound: 0.0002600
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0128037, -0.0111212, -0.0127841, -0.0111237, -0.0009750, 0.0009390
1: -0.0065485, -0.0060741, -0.0065430, -0.0060748, -0.0002749, 0.0002647
2: -0.0097564, -0.0062565, -0.0097157, -0.0062617, -0.0020282, 0.0019533
3: 0.0003362, 0.0007993, 0.0003416, 0.0007987, -0.0002684, 0.0002585
4: 0.0107676, 0.0133832, 0.0107715, 0.0133528, -0.0014598, 0.0015157
5: 0.9984978, 0.9992245, 0.9984989, 0.9992160, -0.0004056, 0.0004211
6: 0.0065201, 0.0071797, 0.0065211, 0.0071720, -0.0003681, 0.0003822
7: 0.0009504, 0.0034119, 0.0009540, 0.0033833, -0.0013738, 0.0014265
8: -0.0118484, -0.0099326, -0.0118261, -0.0099354, -0.0011102, 0.0010692
9: -0.0031528, -0.0029875, -0.0031526, -0.0029894, -0.0000922, 0.0000958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002302, upper bound: 0.0002530
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002477, upper bound: 0.0002550
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128298, -0.0111252, -0.0127841, -0.0111237, -0.0009848, 0.0009120
1: -0.0065559, -0.0060753, -0.0065430, -0.0060748, -0.0002777, 0.0002571
2: -0.0098108, -0.0062648, -0.0097157, -0.0062617, -0.0020486, 0.0018972
3: 0.0003290, 0.0007983, 0.0003416, 0.0007987, -0.0002711, 0.0002511
4: 0.0107738, 0.0134239, 0.0107715, 0.0133528, -0.0014178, 0.0015310
5: 0.9984995, 0.9992357, 0.9984989, 0.9992160, -0.0003939, 0.0004254
6: 0.0065217, 0.0071900, 0.0065211, 0.0071720, -0.0003576, 0.0003861
7: 0.0009562, 0.0034502, 0.0009540, 0.0033833, -0.0013343, 0.0014408
8: -0.0118782, -0.0099371, -0.0118261, -0.0099354, -0.0011214, 0.0010385
9: -0.0031524, -0.0029849, -0.0031526, -0.0029894, -0.0000896, 0.0000967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002302, upper bound: 0.0002589
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002477, upper bound: 0.0002601
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0128900, -0.0111550, -0.0127018, -0.0110922, -0.0010772, 0.0008164
1: -0.0065728, -0.0060837, -0.0065198, -0.0060660, -0.0003037, 0.0002302
2: -0.0099359, -0.0063268, -0.0095445, -0.0061961, -0.0022407, 0.0016984
3: 0.0003124, 0.0007900, 0.0003642, 0.0008073, -0.0002965, 0.0002248
4: 0.0108201, 0.0135174, 0.0107225, 0.0132248, -0.0012693, 0.0016746
5: 0.9985123, 0.9992618, 0.9984852, 0.9991804, -0.0003526, 0.0004652
6: 0.0065333, 0.0072135, 0.0065087, 0.0071398, -0.0003201, 0.0004223
7: 0.0009998, 0.0035382, 0.0009079, 0.0032629, -0.0011945, 0.0015759
8: -0.0119467, -0.0099710, -0.0117324, -0.0098995, -0.0012266, 0.0009297
9: -0.0031495, -0.0029790, -0.0031557, -0.0029975, -0.0000802, 0.0001058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002396, upper bound: 0.0002444
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002441, upper bound: 0.0002637
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0128900, -0.0111550, -0.0127641, -0.0111211, -0.0010328, 0.0008558
1: -0.0065728, -0.0060837, -0.0065373, -0.0060741, -0.0002912, 0.0002413
2: -0.0099359, -0.0063268, -0.0096741, -0.0062564, -0.0021484, 0.0017803
3: 0.0003124, 0.0007900, 0.0003471, 0.0007994, -0.0002843, 0.0002356
4: 0.0108201, 0.0135174, 0.0107675, 0.0133217, -0.0013305, 0.0016056
5: 0.9985123, 0.9992618, 0.9984977, 0.9992074, -0.0003697, 0.0004461
6: 0.0065333, 0.0072135, 0.0065201, 0.0071642, -0.0003355, 0.0004049
7: 0.0009998, 0.0035382, 0.0009503, 0.0033541, -0.0012521, 0.0015110
8: -0.0119467, -0.0099710, -0.0118033, -0.0099325, -0.0011761, 0.0009745
9: -0.0031495, -0.0029790, -0.0031528, -0.0029914, -0.0000841, 0.0001015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002636
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002442, upper bound: 0.0002637
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0129012, -0.0111549, -0.0127273, -0.0110970, -0.0010854, 0.0008277
1: -0.0065760, -0.0060836, -0.0065270, -0.0060673, -0.0003060, 0.0002334
2: -0.0099593, -0.0063266, -0.0095975, -0.0062062, -0.0022578, 0.0017218
3: 0.0003093, 0.0007901, 0.0003572, 0.0008060, -0.0002988, 0.0002278
4: 0.0108200, 0.0135348, 0.0107300, 0.0132644, -0.0012867, 0.0016873
5: 0.9985123, 0.9992666, 0.9984874, 0.9991915, -0.0003575, 0.0004688
6: 0.0065333, 0.0072180, 0.0065106, 0.0071498, -0.0003245, 0.0004255
7: 0.0009997, 0.0035546, 0.0009150, 0.0033002, -0.0012110, 0.0015880
8: -0.0119595, -0.0099709, -0.0117614, -0.0099050, -0.0012359, 0.0009425
9: -0.0031495, -0.0029779, -0.0031552, -0.0029950, -0.0000813, 0.0001066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A2_B2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002482, upper bound: 0.0002608
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002482, upper bound: 0.0002675
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0129012, -0.0111549, -0.0127918, -0.0111233, -0.0010403, 0.0008690
1: -0.0065760, -0.0060836, -0.0065452, -0.0060747, -0.0002933, 0.0002450
2: -0.0099593, -0.0063266, -0.0097317, -0.0062608, -0.0021641, 0.0018077
3: 0.0003093, 0.0007901, 0.0003395, 0.0007988, -0.0002864, 0.0002392
4: 0.0108200, 0.0135348, 0.0107708, 0.0133648, -0.0013509, 0.0016173
5: 0.9985123, 0.9992666, 0.9984987, 0.9992194, -0.0003753, 0.0004493
6: 0.0065333, 0.0072180, 0.0065209, 0.0071751, -0.0003407, 0.0004079
7: 0.0009997, 0.0035546, 0.0009534, 0.0033946, -0.0012714, 0.0015221
8: -0.0119595, -0.0099709, -0.0118349, -0.0099349, -0.0011846, 0.0009895
9: -0.0031495, -0.0029779, -0.0031526, -0.0029887, -0.0000854, 0.0001022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A2_B2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002482, upper bound: 0.0002610
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002482, upper bound: 0.0002676
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0127926, -0.0111215, -0.0128472, -0.0111515, -0.0007523, 0.0008292
1: -0.0065454, -0.0060742, -0.0065608, -0.0060827, -0.0002121, 0.0002338
2: -0.0097334, -0.0062571, -0.0098469, -0.0063195, -0.0015649, 0.0017249
3: 0.0003392, 0.0007993, 0.0003242, 0.0007910, -0.0002071, 0.0002283
4: 0.0107680, 0.0133660, 0.0108146, 0.0134508, -0.0012891, 0.0011695
5: 0.9984980, 0.9992197, 0.9985108, 0.9992433, -0.0003582, 0.0003249
6: 0.0065202, 0.0071754, 0.0065320, 0.0071968, -0.0003251, 0.0002949
7: 0.0009508, 0.0033958, 0.0009946, 0.0034756, -0.0012132, 0.0011006
8: -0.0118358, -0.0099329, -0.0118979, -0.0099670, -0.0008566, 0.0009442
9: -0.0031528, -0.0029886, -0.0031498, -0.0029832, -0.0000815, 0.0000739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002297, upper bound: 0.0002537
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002297, upper bound: 0.0002537
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0128037, -0.0111212, -0.0128779, -0.0111561, -0.0007641, 0.0008366
1: -0.0065485, -0.0060741, -0.0065694, -0.0060840, -0.0002154, 0.0002359
2: -0.0097564, -0.0062565, -0.0099108, -0.0063291, -0.0015895, 0.0017402
3: 0.0003362, 0.0007993, 0.0003158, 0.0007897, -0.0002103, 0.0002303
4: 0.0107676, 0.0133832, 0.0108218, 0.0134985, -0.0013005, 0.0011879
5: 0.9984978, 0.9992245, 0.9985128, 0.9992566, -0.0003613, 0.0003300
6: 0.0065201, 0.0071797, 0.0065338, 0.0072088, -0.0003280, 0.0002996
7: 0.0009504, 0.0034119, 0.0010014, 0.0035205, -0.0012240, 0.0011179
8: -0.0118484, -0.0099326, -0.0119329, -0.0099723, -0.0008701, 0.0009526
9: -0.0031528, -0.0029875, -0.0031494, -0.0029802, -0.0000822, 0.0000751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002518, upper bound: 0.0002552
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002518, upper bound: 0.0002552
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0128298, -0.0111252, -0.0128572, -0.0111498, -0.0007957, 0.0008370
1: -0.0065559, -0.0060753, -0.0065636, -0.0060822, -0.0002243, 0.0002360
2: -0.0098108, -0.0062648, -0.0098678, -0.0063160, -0.0016553, 0.0017411
3: 0.0003290, 0.0007983, 0.0003215, 0.0007915, -0.0002190, 0.0002304
4: 0.0107738, 0.0134239, 0.0108121, 0.0134664, -0.0013012, 0.0012371
5: 0.9984995, 0.9992357, 0.9985102, 0.9992476, -0.0003615, 0.0003437
6: 0.0065217, 0.0071900, 0.0065313, 0.0072007, -0.0003282, 0.0003120
7: 0.0009562, 0.0034502, 0.0009922, 0.0034903, -0.0012246, 0.0011642
8: -0.0118782, -0.0099371, -0.0119094, -0.0099651, -0.0009061, 0.0009531
9: -0.0031524, -0.0029849, -0.0031500, -0.0029823, -0.0000822, 0.0000782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002326, upper bound: 0.0002592
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002478, upper bound: 0.0002602
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0128298, -0.0111252, -0.0128834, -0.0111556, -0.0007692, 0.0008491
1: -0.0065559, -0.0060753, -0.0065710, -0.0060838, -0.0002169, 0.0002394
2: -0.0098108, -0.0062648, -0.0099223, -0.0063281, -0.0016001, 0.0017662
3: 0.0003290, 0.0007983, 0.0003142, 0.0007899, -0.0002117, 0.0002337
4: 0.0107738, 0.0134239, 0.0108211, 0.0135072, -0.0013200, 0.0011958
5: 0.9984995, 0.9992357, 0.9985127, 0.9992590, -0.0003667, 0.0003322
6: 0.0065217, 0.0071900, 0.0065336, 0.0072110, -0.0003329, 0.0003016
7: 0.0009562, 0.0034502, 0.0010007, 0.0035286, -0.0012422, 0.0011254
8: -0.0118782, -0.0099371, -0.0119392, -0.0099717, -0.0008759, 0.0009668
9: -0.0031524, -0.0029849, -0.0031494, -0.0029797, -0.0000834, 0.0000756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002326, upper bound: 0.0002593
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002478, upper bound: 0.0002602
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0128615, -0.0111492, -0.0128327, -0.0111251, -0.0008650, 0.0007908
1: -0.0065648, -0.0060820, -0.0065567, -0.0060752, -0.0002439, 0.0002230
2: -0.0098766, -0.0063148, -0.0098167, -0.0062646, -0.0017993, 0.0016451
3: 0.0003203, 0.0007916, 0.0003282, 0.0007983, -0.0002381, 0.0002177
4: 0.0108112, 0.0134731, 0.0107737, 0.0134283, -0.0012295, 0.0013447
5: 0.9985099, 0.9992495, 0.9984995, 0.9992370, -0.0003416, 0.0003736
6: 0.0065311, 0.0072024, 0.0065216, 0.0071911, -0.0003101, 0.0003391
7: 0.0009914, 0.0034965, 0.0009561, 0.0034543, -0.0011571, 0.0012655
8: -0.0119142, -0.0099645, -0.0118814, -0.0099370, -0.0009849, 0.0009005
9: -0.0031501, -0.0029818, -0.0031524, -0.0029847, -0.0000777, 0.0000850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002490, upper bound: 0.0002364
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002498, upper bound: 0.0002575
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0128615, -0.0111492, -0.0128900, -0.0111550, -0.0007916, 0.0008202
1: -0.0065648, -0.0060820, -0.0065728, -0.0060837, -0.0002232, 0.0002313
2: -0.0098766, -0.0063148, -0.0099359, -0.0063268, -0.0016468, 0.0017062
3: 0.0003203, 0.0007916, 0.0003124, 0.0007900, -0.0002179, 0.0002258
4: 0.0108112, 0.0134731, 0.0108201, 0.0135174, -0.0012751, 0.0012307
5: 0.9985099, 0.9992495, 0.9985123, 0.9992618, -0.0003543, 0.0003419
6: 0.0065311, 0.0072024, 0.0065333, 0.0072135, -0.0003216, 0.0003104
7: 0.0009914, 0.0034965, 0.0009998, 0.0035382, -0.0012000, 0.0011582
8: -0.0119142, -0.0099645, -0.0119467, -0.0099710, -0.0009014, 0.0009340
9: -0.0031501, -0.0029818, -0.0031495, -0.0029790, -0.0000806, 0.0000778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002294, upper bound: 0.0002575
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002499, upper bound: 0.0002576
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0128873, -0.0111551, -0.0128437, -0.0111250, -0.0008766, 0.0008024
1: -0.0065721, -0.0060837, -0.0065598, -0.0060752, -0.0002471, 0.0002262
2: -0.0099304, -0.0063269, -0.0098396, -0.0062645, -0.0018235, 0.0016691
3: 0.0003132, 0.0007900, 0.0003252, 0.0007983, -0.0002413, 0.0002209
4: 0.0108202, 0.0135132, 0.0107735, 0.0134454, -0.0012473, 0.0013628
5: 0.9985124, 0.9992606, 0.9984994, 0.9992418, -0.0003465, 0.0003786
6: 0.0065334, 0.0072125, 0.0065216, 0.0071954, -0.0003146, 0.0003437
7: 0.0009999, 0.0035343, 0.0009560, 0.0034705, -0.0011739, 0.0012825
8: -0.0119436, -0.0099711, -0.0118940, -0.0099369, -0.0009982, 0.0009136
9: -0.0031495, -0.0029793, -0.0031524, -0.0029836, -0.0000788, 0.0000861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002485, upper bound: 0.0002675
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002485, upper bound: 0.0002676
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0128873, -0.0111551, -0.0129012, -0.0111549, -0.0008049, 0.0008298
1: -0.0065721, -0.0060837, -0.0065760, -0.0060836, -0.0002269, 0.0002339
2: -0.0099304, -0.0063269, -0.0099593, -0.0063266, -0.0016744, 0.0017261
3: 0.0003132, 0.0007900, 0.0003093, 0.0007901, -0.0002216, 0.0002284
4: 0.0108202, 0.0135132, 0.0108200, 0.0135348, -0.0012900, 0.0012514
5: 0.9985124, 0.9992606, 0.9985123, 0.9992666, -0.0003584, 0.0003477
6: 0.0065334, 0.0072125, 0.0065333, 0.0072180, -0.0003253, 0.0003156
7: 0.0009999, 0.0035343, 0.0009997, 0.0035546, -0.0012140, 0.0011777
8: -0.0119436, -0.0099711, -0.0119595, -0.0099709, -0.0009166, 0.0009449
9: -0.0031495, -0.0029793, -0.0031495, -0.0029779, -0.0000815, 0.0000791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002485, upper bound: 0.0002677
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002485, upper bound: 0.0002678
time: 0.62 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.16 seconds
IS_A1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002397
IS_A1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002397
IS_A1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002514, upper bound: 0.0002448
IS_A1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002514, upper bound: 0.0002448
IS_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002303, upper bound: 0.0002472
IS_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002477, upper bound: 0.0002500
IS_A1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002303, upper bound: 0.0002480
IS_A1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002477, upper bound: 0.0002501
IS_A1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002472, upper bound: 0.0002303
IS_A1_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002501, upper bound: 0.0002478
IS_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002479
IS_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002501, upper bound: 0.0002479
IS_A1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002488, upper bound: 0.0002590
IS_A1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002488, upper bound: 0.0002591
IS_A1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002488, upper bound: 0.0002591
IS_A1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002488, upper bound: 0.0002594
IS_A1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002585, upper bound: 0.0002201
IS_A1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002585, upper bound: 0.0002303
IS_A1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002599, upper bound: 0.0002446
IS_A1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002599, upper bound: 0.0002477
IS_A1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002529, upper bound: 0.0002413
IS_A1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002549, upper bound: 0.0002551
IS_A1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002529, upper bound: 0.0002418
IS_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002549, upper bound: 0.0002553
IS_A1_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002321, upper bound: 0.0002396
IS_A1_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002598, upper bound: 0.0002442
IS_A1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002585, upper bound: 0.0002349
IS_A1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002600, upper bound: 0.0002480
IS_A1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002587, upper bound: 0.0002536
IS_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002587, upper bound: 0.0002537
IS_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002587, upper bound: 0.0002592
IS_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002587, upper bound: 0.0002593
IS_A2_B1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002585
IS_A2_B1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002585
IS_A2_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002446, upper bound: 0.0002600
IS_A2_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002446, upper bound: 0.0002600
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002302, upper bound: 0.0002530
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002477, upper bound: 0.0002550
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002302, upper bound: 0.0002589
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002477, upper bound: 0.0002601
IS_A2_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002396, upper bound: 0.0002444
IS_A2_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002441, upper bound: 0.0002637
IS_A2_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002636
IS_A2_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002442, upper bound: 0.0002637
IS_A2_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002482, upper bound: 0.0002608
IS_A2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002482, upper bound: 0.0002675
IS_A2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002482, upper bound: 0.0002610
IS_A2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002482, upper bound: 0.0002676
IS_A2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002297, upper bound: 0.0002537
IS_A2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002297, upper bound: 0.0002537
IS_A2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002518, upper bound: 0.0002552
IS_A2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002518, upper bound: 0.0002552
IS_A2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002326, upper bound: 0.0002592
IS_A2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002478, upper bound: 0.0002602
IS_A2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002326, upper bound: 0.0002593
IS_A2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002478, upper bound: 0.0002602
IS_A2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002490, upper bound: 0.0002364
IS_A2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002498, upper bound: 0.0002575
IS_A2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002294, upper bound: 0.0002575
IS_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002499, upper bound: 0.0002576
IS_A2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002485, upper bound: 0.0002675
IS_A2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002485, upper bound: 0.0002676
IS_A2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002485, upper bound: 0.0002677
IS_A2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 5, lower bound: -0.0002485, upper bound: 0.0002678

## BFS IS instance: IS_A1_B1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0126913, -0.0110924, -0.0126935, -0.0110903, -0.0007437, 0.0007409
1: -0.0065168, -0.0060660, -0.0065174, -0.0060654, -0.0002097, 0.0002089
2: -0.0095226, -0.0061967, -0.0095272, -0.0061922, -0.0015471, 0.0015412
3: 0.0003671, 0.0008073, 0.0003665, 0.0008079, -0.0002047, 0.0002039
4: 0.0107229, 0.0132085, 0.0107195, 0.0132119, -0.0011518, 0.0011562
5: 0.9984854, 0.9991759, 0.9984844, 0.9991769, -0.0003200, 0.0003212
6: 0.0065088, 0.0071357, 0.0065080, 0.0071365, -0.0002905, 0.0002916
7: 0.0009083, 0.0032475, 0.0009052, 0.0032507, -0.0010839, 0.0010881
8: -0.0117204, -0.0098998, -0.0117229, -0.0098973, -0.0008469, 0.0008436
9: -0.0031556, -0.0029986, -0.0031558, -0.0029983, -0.0000728, 0.0000731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002205, upper bound: 0.0002397
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002205, upper bound: 0.0002397
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0126913, -0.0110924, -0.0127568, -0.0111155, -0.0007352, 0.0008351
1: -0.0065168, -0.0060660, -0.0065353, -0.0060725, -0.0002073, 0.0002354
2: -0.0095226, -0.0061967, -0.0096589, -0.0062446, -0.0015294, 0.0017371
3: 0.0003671, 0.0008073, 0.0003491, 0.0008009, -0.0002024, 0.0002299
4: 0.0107229, 0.0132085, 0.0107587, 0.0133103, -0.0012982, 0.0011429
5: 0.9984854, 0.9991759, 0.9984953, 0.9992042, -0.0003607, 0.0003175
6: 0.0065088, 0.0071357, 0.0065178, 0.0071613, -0.0003274, 0.0002882
7: 0.0009083, 0.0032475, 0.0009420, 0.0033433, -0.0012217, 0.0010756
8: -0.0117204, -0.0098998, -0.0117950, -0.0099260, -0.0008372, 0.0009509
9: -0.0031556, -0.0029986, -0.0031534, -0.0029921, -0.0000820, 0.0000722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002205, upper bound: 0.0002397
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002205, upper bound: 0.0002397
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0127018, -0.0110922, -0.0127211, -0.0110974, -0.0007545, 0.0007467
1: -0.0065198, -0.0060660, -0.0065252, -0.0060674, -0.0002127, 0.0002105
2: -0.0095445, -0.0061961, -0.0095846, -0.0062070, -0.0015696, 0.0015532
3: 0.0003642, 0.0008073, 0.0003589, 0.0008059, -0.0002077, 0.0002055
4: 0.0107225, 0.0132248, 0.0107306, 0.0132548, -0.0011608, 0.0011730
5: 0.9984852, 0.9991804, 0.9984875, 0.9991888, -0.0003225, 0.0003259
6: 0.0065087, 0.0071398, 0.0065108, 0.0071473, -0.0002927, 0.0002958
7: 0.0009079, 0.0032629, 0.0009156, 0.0032911, -0.0010924, 0.0011039
8: -0.0117324, -0.0098995, -0.0117544, -0.0099055, -0.0008592, 0.0008502
9: -0.0031557, -0.0029975, -0.0031551, -0.0029956, -0.0000734, 0.0000741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002201
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002447
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0127018, -0.0110922, -0.0127847, -0.0111238, -0.0007459, 0.0008408
1: -0.0065198, -0.0060660, -0.0065431, -0.0060749, -0.0002103, 0.0002371
2: -0.0095445, -0.0061961, -0.0097170, -0.0062619, -0.0015516, 0.0017491
3: 0.0003642, 0.0008073, 0.0003414, 0.0007986, -0.0002053, 0.0002315
4: 0.0107225, 0.0132248, 0.0107716, 0.0133537, -0.0013072, 0.0011596
5: 0.9984852, 0.9991804, 0.9984989, 0.9992163, -0.0003632, 0.0003222
6: 0.0065087, 0.0071398, 0.0065211, 0.0071723, -0.0003296, 0.0002924
7: 0.0009079, 0.0032629, 0.0009542, 0.0033842, -0.0012302, 0.0010913
8: -0.0117324, -0.0098995, -0.0118268, -0.0099355, -0.0008494, 0.0009575
9: -0.0031557, -0.0029975, -0.0031525, -0.0029894, -0.0000826, 0.0000733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002201
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002448
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0127168, -0.0110973, -0.0127209, -0.0111130, -0.0007671, 0.0007744
1: -0.0065240, -0.0060674, -0.0065252, -0.0060718, -0.0002163, 0.0002183
2: -0.0095757, -0.0062067, -0.0095843, -0.0062394, -0.0015957, 0.0016109
3: 0.0003601, 0.0008059, 0.0003590, 0.0008016, -0.0002112, 0.0002132
4: 0.0107304, 0.0132482, 0.0107548, 0.0132546, -0.0012039, 0.0011925
5: 0.9984875, 0.9991870, 0.9984943, 0.9991887, -0.0003345, 0.0003313
6: 0.0065107, 0.0071457, 0.0065169, 0.0071473, -0.0003036, 0.0003007
7: 0.0009153, 0.0032849, 0.0009383, 0.0032909, -0.0011330, 0.0011223
8: -0.0117495, -0.0099053, -0.0117542, -0.0099232, -0.0008735, 0.0008818
9: -0.0031552, -0.0029960, -0.0031536, -0.0029956, -0.0000761, 0.0000754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002472
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002472
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0127273, -0.0110970, -0.0127483, -0.0111222, -0.0007772, 0.0007821
1: -0.0065270, -0.0060673, -0.0065329, -0.0060744, -0.0002191, 0.0002205
2: -0.0095975, -0.0062062, -0.0096412, -0.0062586, -0.0016166, 0.0016269
3: 0.0003572, 0.0008060, 0.0003514, 0.0007991, -0.0002139, 0.0002153
4: 0.0107300, 0.0132644, 0.0107692, 0.0132971, -0.0012158, 0.0012082
5: 0.9984874, 0.9991915, 0.9984982, 0.9992006, -0.0003378, 0.0003357
6: 0.0065106, 0.0071498, 0.0065205, 0.0071580, -0.0003066, 0.0003047
7: 0.0009150, 0.0033002, 0.0009518, 0.0033309, -0.0011442, 0.0011370
8: -0.0117614, -0.0099050, -0.0117853, -0.0099337, -0.0008849, 0.0008906
9: -0.0031552, -0.0029950, -0.0031527, -0.0029930, -0.0000768, 0.0000763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002449, upper bound: 0.0002501
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002449, upper bound: 0.0002502
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0127168, -0.0110973, -0.0127478, -0.0111159, -0.0007398, 0.0007866
1: -0.0065240, -0.0060674, -0.0065327, -0.0060727, -0.0002086, 0.0002218
2: -0.0095757, -0.0062067, -0.0096401, -0.0062456, -0.0015389, 0.0016362
3: 0.0003601, 0.0008059, 0.0003516, 0.0008008, -0.0002037, 0.0002165
4: 0.0107304, 0.0132482, 0.0107594, 0.0132963, -0.0012228, 0.0011501
5: 0.9984875, 0.9991870, 0.9984955, 0.9992003, -0.0003397, 0.0003195
6: 0.0065107, 0.0071457, 0.0065180, 0.0071578, -0.0003084, 0.0002900
7: 0.0009153, 0.0032849, 0.0009427, 0.0033301, -0.0011508, 0.0010824
8: -0.0117495, -0.0099053, -0.0117847, -0.0099266, -0.0008424, 0.0008957
9: -0.0031552, -0.0029960, -0.0031533, -0.0029930, -0.0000773, 0.0000727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002480
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002480
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0127273, -0.0110970, -0.0127753, -0.0111243, -0.0007506, 0.0007922
1: -0.0065270, -0.0060673, -0.0065405, -0.0060750, -0.0002116, 0.0002233
2: -0.0095975, -0.0062062, -0.0096974, -0.0062630, -0.0015615, 0.0016479
3: 0.0003572, 0.0008060, 0.0003440, 0.0007985, -0.0002066, 0.0002181
4: 0.0107300, 0.0132644, 0.0107724, 0.0133391, -0.0012315, 0.0011669
5: 0.9984874, 0.9991915, 0.9984992, 0.9992123, -0.0003422, 0.0003242
6: 0.0065106, 0.0071498, 0.0065213, 0.0071686, -0.0003106, 0.0002943
7: 0.0009150, 0.0033002, 0.0009549, 0.0033704, -0.0011590, 0.0010982
8: -0.0117614, -0.0099050, -0.0118161, -0.0099361, -0.0008547, 0.0009021
9: -0.0031552, -0.0029950, -0.0031525, -0.0029903, -0.0000778, 0.0000737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002449, upper bound: 0.0002502
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002449, upper bound: 0.0002502
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0127277, -0.0111125, -0.0127195, -0.0110972, -0.0008045, 0.0007606
1: -0.0065271, -0.0060717, -0.0065248, -0.0060674, -0.0002268, 0.0002144
2: -0.0095983, -0.0062385, -0.0095812, -0.0062066, -0.0016735, 0.0015822
3: 0.0003571, 0.0008017, 0.0003594, 0.0008060, -0.0002215, 0.0002094
4: 0.0107541, 0.0132650, 0.0107303, 0.0132523, -0.0011824, 0.0012506
5: 0.9984940, 0.9991917, 0.9984874, 0.9991882, -0.0003285, 0.0003475
6: 0.0065167, 0.0071499, 0.0065107, 0.0071467, -0.0002982, 0.0003154
7: 0.0009377, 0.0033007, 0.0009152, 0.0032887, -0.0011128, 0.0011770
8: -0.0117618, -0.0099227, -0.0117525, -0.0099052, -0.0009161, 0.0008661
9: -0.0031537, -0.0029950, -0.0031552, -0.0029958, -0.0000747, 0.0000790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002421, upper bound: 0.0002303
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002421, upper bound: 0.0002303
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127554, -0.0111218, -0.0127300, -0.0110969, -0.0008124, 0.0007714
1: -0.0065349, -0.0060743, -0.0065277, -0.0060673, -0.0002290, 0.0002175
2: -0.0096560, -0.0062577, -0.0096031, -0.0062060, -0.0016899, 0.0016048
3: 0.0003495, 0.0007992, 0.0003565, 0.0008060, -0.0002236, 0.0002124
4: 0.0107685, 0.0133081, 0.0107299, 0.0132686, -0.0011993, 0.0012629
5: 0.9984980, 0.9992037, 0.9984873, 0.9991927, -0.0003332, 0.0003509
6: 0.0065203, 0.0071608, 0.0065106, 0.0071508, -0.0003024, 0.0003185
7: 0.0009512, 0.0033413, 0.0009149, 0.0033041, -0.0011287, 0.0011885
8: -0.0117934, -0.0099332, -0.0117644, -0.0099049, -0.0009250, 0.0008785
9: -0.0031527, -0.0029923, -0.0031552, -0.0029948, -0.0000758, 0.0000798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002473
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002478
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0127536, -0.0111215, -0.0127568, -0.0111155, -0.0007616, 0.0007588
1: -0.0065344, -0.0060742, -0.0065353, -0.0060725, -0.0002147, 0.0002139
2: -0.0096523, -0.0062571, -0.0096589, -0.0062446, -0.0015843, 0.0015785
3: 0.0003500, 0.0007993, 0.0003491, 0.0008009, -0.0002097, 0.0002089
4: 0.0107681, 0.0133054, 0.0107587, 0.0133103, -0.0011797, 0.0011840
5: 0.9984980, 0.9992029, 0.9984953, 0.9992042, -0.0003278, 0.0003290
6: 0.0065202, 0.0071601, 0.0065178, 0.0071613, -0.0002975, 0.0002986
7: 0.0009508, 0.0033387, 0.0009420, 0.0033433, -0.0011102, 0.0011143
8: -0.0117914, -0.0099329, -0.0117950, -0.0099260, -0.0008673, 0.0008641
9: -0.0031528, -0.0029924, -0.0031534, -0.0029921, -0.0000745, 0.0000748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002327, upper bound: 0.0002478
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002327, upper bound: 0.0002479
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0127641, -0.0111211, -0.0127847, -0.0111238, -0.0007725, 0.0007658
1: -0.0065373, -0.0060741, -0.0065431, -0.0060749, -0.0002178, 0.0002159
2: -0.0096741, -0.0062564, -0.0097170, -0.0062619, -0.0016069, 0.0015930
3: 0.0003471, 0.0007994, 0.0003414, 0.0007986, -0.0002126, 0.0002108
4: 0.0107675, 0.0133217, 0.0107716, 0.0133537, -0.0011905, 0.0012009
5: 0.9984977, 0.9992074, 0.9984989, 0.9992163, -0.0003308, 0.0003336
6: 0.0065201, 0.0071642, 0.0065211, 0.0071723, -0.0003002, 0.0003028
7: 0.0009503, 0.0033541, 0.0009542, 0.0033842, -0.0011204, 0.0011301
8: -0.0118033, -0.0099325, -0.0118268, -0.0099355, -0.0008796, 0.0008720
9: -0.0031528, -0.0029914, -0.0031525, -0.0029894, -0.0000752, 0.0000759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002503, upper bound: 0.0002349
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002503, upper bound: 0.0002480
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0127918, -0.0111233, -0.0127018, -0.0110922, -0.0008804, 0.0007421
1: -0.0065452, -0.0060747, -0.0065198, -0.0060660, -0.0002482, 0.0002092
2: -0.0097317, -0.0062608, -0.0095445, -0.0061961, -0.0018315, 0.0015438
3: 0.0003395, 0.0007988, 0.0003642, 0.0008073, -0.0002424, 0.0002043
4: 0.0107708, 0.0133648, 0.0107225, 0.0132248, -0.0011537, 0.0013687
5: 0.9984987, 0.9992194, 0.9984852, 0.9991804, -0.0003205, 0.0003803
6: 0.0065209, 0.0071751, 0.0065087, 0.0071398, -0.0002910, 0.0003452
7: 0.0009534, 0.0033946, 0.0009079, 0.0032629, -0.0010858, 0.0012881
8: -0.0118349, -0.0099349, -0.0117324, -0.0098995, -0.0010025, 0.0008451
9: -0.0031526, -0.0029887, -0.0031557, -0.0029975, -0.0000729, 0.0000865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002397, upper bound: 0.0002413
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002447, upper bound: 0.0002552
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0127918, -0.0111233, -0.0127273, -0.0110970, -0.0008526, 0.0007510
1: -0.0065452, -0.0060747, -0.0065270, -0.0060673, -0.0002404, 0.0002117
2: -0.0097317, -0.0062608, -0.0095975, -0.0062062, -0.0017735, 0.0015622
3: 0.0003395, 0.0007988, 0.0003572, 0.0008060, -0.0002347, 0.0002067
4: 0.0107708, 0.0133648, 0.0107300, 0.0132644, -0.0011675, 0.0013254
5: 0.9984987, 0.9992194, 0.9984874, 0.9991915, -0.0003244, 0.0003682
6: 0.0065209, 0.0071751, 0.0065106, 0.0071498, -0.0002944, 0.0003342
7: 0.0009534, 0.0033946, 0.0009150, 0.0033002, -0.0010988, 0.0012473
8: -0.0118349, -0.0099349, -0.0117614, -0.0099050, -0.0009708, 0.0008552
9: -0.0031526, -0.0029887, -0.0031552, -0.0029950, -0.0000738, 0.0000838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002397, upper bound: 0.0002418
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002447, upper bound: 0.0002552
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0127918, -0.0111233, -0.0127641, -0.0111211, -0.0008063, 0.0007684
1: -0.0065452, -0.0060747, -0.0065373, -0.0060741, -0.0002273, 0.0002167
2: -0.0097317, -0.0062608, -0.0096741, -0.0062564, -0.0016773, 0.0015985
3: 0.0003395, 0.0007988, 0.0003471, 0.0007994, -0.0002220, 0.0002115
4: 0.0107708, 0.0133648, 0.0107675, 0.0133217, -0.0011946, 0.0012535
5: 0.9984987, 0.9992194, 0.9984977, 0.9992074, -0.0003319, 0.0003483
6: 0.0065209, 0.0071751, 0.0065201, 0.0071642, -0.0003013, 0.0003161
7: 0.0009534, 0.0033946, 0.0009503, 0.0033541, -0.0011243, 0.0011797
8: -0.0118349, -0.0099349, -0.0118033, -0.0099325, -0.0009181, 0.0008750
9: -0.0031526, -0.0029887, -0.0031528, -0.0029914, -0.0000755, 0.0000792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002448, upper bound: 0.0002426
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002450, upper bound: 0.0002553
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0127918, -0.0111233, -0.0127918, -0.0111233, -0.0007816, 0.0007816
1: -0.0065452, -0.0060747, -0.0065452, -0.0060747, -0.0002204, 0.0002204
2: -0.0097317, -0.0062608, -0.0097317, -0.0062608, -0.0016258, 0.0016258
3: 0.0003395, 0.0007988, 0.0003395, 0.0007988, -0.0002152, 0.0002152
4: 0.0107708, 0.0133648, 0.0107708, 0.0133648, -0.0012151, 0.0012151
5: 0.9984987, 0.9992194, 0.9984987, 0.9992194, -0.0003376, 0.0003376
6: 0.0065209, 0.0071751, 0.0065209, 0.0071751, -0.0003064, 0.0003064
7: 0.0009534, 0.0033946, 0.0009534, 0.0033946, -0.0011435, 0.0011435
8: -0.0118349, -0.0099349, -0.0118349, -0.0099349, -0.0008900, 0.0008900
9: -0.0031526, -0.0029887, -0.0031526, -0.0029887, -0.0000768, 0.0000768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002448, upper bound: 0.0002428
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002450, upper bound: 0.0002556
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0126650, -0.0110851, -0.0128216, -0.0111254, -0.0007951, 0.0010040
1: -0.0065094, -0.0060640, -0.0065535, -0.0060753, -0.0002242, 0.0002831
2: -0.0094679, -0.0061813, -0.0097936, -0.0062652, -0.0016539, 0.0020885
3: 0.0003744, 0.0008093, 0.0003313, 0.0007982, -0.0002189, 0.0002764
4: 0.0107114, 0.0131676, 0.0107741, 0.0134110, -0.0015608, 0.0012360
5: 0.9984822, 0.9991646, 0.9984996, 0.9992322, -0.0004336, 0.0003434
6: 0.0065059, 0.0071253, 0.0065217, 0.0071867, -0.0003936, 0.0003117
7: 0.0008975, 0.0032090, 0.0009565, 0.0034381, -0.0014689, 0.0011632
8: -0.0116905, -0.0098914, -0.0118688, -0.0099373, -0.0009053, 0.0011432
9: -0.0031564, -0.0030011, -0.0031524, -0.0029858, -0.0000986, 0.0000781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002545, upper bound: 0.0002201
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002545, upper bound: 0.0002201
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0127277, -0.0111125, -0.0128216, -0.0111254, -0.0008873, 0.0009948
1: -0.0065271, -0.0060717, -0.0065535, -0.0060753, -0.0002502, 0.0002805
2: -0.0095983, -0.0062385, -0.0097936, -0.0062652, -0.0018457, 0.0020693
3: 0.0003571, 0.0008017, 0.0003313, 0.0007982, -0.0002442, 0.0002738
4: 0.0107541, 0.0132650, 0.0107741, 0.0134110, -0.0015465, 0.0013794
5: 0.9984940, 0.9991917, 0.9984996, 0.9992322, -0.0004297, 0.0003832
6: 0.0065167, 0.0071499, 0.0065217, 0.0071867, -0.0003900, 0.0003479
7: 0.0009377, 0.0033007, 0.0009565, 0.0034381, -0.0014554, 0.0012981
8: -0.0117618, -0.0099227, -0.0118688, -0.0099373, -0.0010103, 0.0011328
9: -0.0031537, -0.0029950, -0.0031524, -0.0029858, -0.0000977, 0.0000872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002545, upper bound: 0.0002303
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002545, upper bound: 0.0002303
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0126930, -0.0110927, -0.0128327, -0.0111251, -0.0008052, 0.0010152
1: -0.0065173, -0.0060661, -0.0065567, -0.0060752, -0.0002270, 0.0002862
2: -0.0095261, -0.0061972, -0.0098167, -0.0062646, -0.0016750, 0.0021117
3: 0.0003667, 0.0008072, 0.0003282, 0.0007983, -0.0002217, 0.0002795
4: 0.0107232, 0.0132111, 0.0107737, 0.0134283, -0.0015782, 0.0012518
5: 0.9984855, 0.9991767, 0.9984995, 0.9992370, -0.0004385, 0.0003478
6: 0.0065089, 0.0071363, 0.0065216, 0.0071911, -0.0003980, 0.0003157
7: 0.0009086, 0.0032499, 0.0009561, 0.0034543, -0.0014852, 0.0011780
8: -0.0117223, -0.0099001, -0.0118814, -0.0099370, -0.0009169, 0.0011560
9: -0.0031556, -0.0029984, -0.0031524, -0.0029847, -0.0000997, 0.0000791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002321, upper bound: 0.0002396
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002321, upper bound: 0.0002445
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0127554, -0.0111218, -0.0128327, -0.0111251, -0.0008974, 0.0010040
1: -0.0065349, -0.0060743, -0.0065567, -0.0060752, -0.0002530, 0.0002831
2: -0.0096560, -0.0062577, -0.0098167, -0.0062646, -0.0018667, 0.0020885
3: 0.0003495, 0.0007992, 0.0003282, 0.0007983, -0.0002470, 0.0002764
4: 0.0107685, 0.0133081, 0.0107737, 0.0134283, -0.0015609, 0.0013951
5: 0.9984980, 0.9992037, 0.9984995, 0.9992370, -0.0004337, 0.0003876
6: 0.0065203, 0.0071608, 0.0065216, 0.0071911, -0.0003936, 0.0003518
7: 0.0009512, 0.0033413, 0.0009561, 0.0034543, -0.0014689, 0.0013129
8: -0.0117934, -0.0099332, -0.0118814, -0.0099370, -0.0010218, 0.0011433
9: -0.0031527, -0.0029923, -0.0031524, -0.0029847, -0.0000986, 0.0000882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002321, upper bound: 0.0002472
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002321, upper bound: 0.0002477
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0127478, -0.0111159, -0.0127926, -0.0111215, -0.0008984, 0.0009656
1: -0.0065327, -0.0060727, -0.0065454, -0.0060742, -0.0002533, 0.0002722
2: -0.0096401, -0.0062456, -0.0097334, -0.0062571, -0.0018689, 0.0020086
3: 0.0003516, 0.0008008, 0.0003392, 0.0007993, -0.0002473, 0.0002658
4: 0.0107594, 0.0132963, 0.0107680, 0.0133660, -0.0015011, 0.0013967
5: 0.9984955, 0.9992003, 0.9984980, 0.9992197, -0.0004171, 0.0003881
6: 0.0065180, 0.0071578, 0.0065202, 0.0071754, -0.0003786, 0.0003522
7: 0.0009427, 0.0033301, 0.0009508, 0.0033958, -0.0014127, 0.0013145
8: -0.0117847, -0.0099266, -0.0118358, -0.0099329, -0.0010231, 0.0010995
9: -0.0031533, -0.0029930, -0.0031528, -0.0029886, -0.0000949, 0.0000883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002530, upper bound: 0.0002255
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002530, upper bound: 0.0002413
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127753, -0.0111243, -0.0128037, -0.0111212, -0.0009107, 0.0009744
1: -0.0065405, -0.0060750, -0.0065485, -0.0060741, -0.0002568, 0.0002747
2: -0.0096974, -0.0062630, -0.0097564, -0.0062565, -0.0018944, 0.0020269
3: 0.0003440, 0.0007985, 0.0003362, 0.0007993, -0.0002507, 0.0002682
4: 0.0107724, 0.0133391, 0.0107676, 0.0133832, -0.0015148, 0.0014157
5: 0.9984992, 0.9992123, 0.9984978, 0.9992245, -0.0004209, 0.0003933
6: 0.0065213, 0.0071686, 0.0065201, 0.0071797, -0.0003820, 0.0003570
7: 0.0009549, 0.0033704, 0.0009504, 0.0034119, -0.0014256, 0.0013324
8: -0.0118161, -0.0099361, -0.0118484, -0.0099326, -0.0010370, 0.0011095
9: -0.0031525, -0.0029903, -0.0031528, -0.0029875, -0.0000957, 0.0000895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002550, upper bound: 0.0002514
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002550, upper bound: 0.0002551
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0127478, -0.0111159, -0.0128187, -0.0111255, -0.0008698, 0.0009752
1: -0.0065327, -0.0060727, -0.0065527, -0.0060753, -0.0002452, 0.0002750
2: -0.0096401, -0.0062456, -0.0097876, -0.0062654, -0.0018094, 0.0020287
3: 0.0003516, 0.0008008, 0.0003321, 0.0007982, -0.0002394, 0.0002685
4: 0.0107594, 0.0132963, 0.0107742, 0.0134065, -0.0015161, 0.0013522
5: 0.9984955, 0.9992003, 0.9984996, 0.9992309, -0.0004212, 0.0003757
6: 0.0065180, 0.0071578, 0.0065218, 0.0071856, -0.0003823, 0.0003410
7: 0.0009427, 0.0033301, 0.0009566, 0.0034339, -0.0014268, 0.0012726
8: -0.0117847, -0.0099266, -0.0118654, -0.0099374, -0.0009905, 0.0011105
9: -0.0031533, -0.0029930, -0.0031524, -0.0029860, -0.0000958, 0.0000855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002542, upper bound: 0.0002275
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002542, upper bound: 0.0002417
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127753, -0.0111243, -0.0128298, -0.0111252, -0.0008824, 0.0009842
1: -0.0065405, -0.0060750, -0.0065559, -0.0060753, -0.0002488, 0.0002775
2: -0.0096974, -0.0062630, -0.0098108, -0.0062648, -0.0018355, 0.0020474
3: 0.0003440, 0.0007985, 0.0003290, 0.0007983, -0.0002429, 0.0002709
4: 0.0107724, 0.0133391, 0.0107738, 0.0134239, -0.0015301, 0.0013717
5: 0.9984992, 0.9992123, 0.9984995, 0.9992357, -0.0004251, 0.0003811
6: 0.0065213, 0.0071686, 0.0065217, 0.0071900, -0.0003859, 0.0003459
7: 0.0009549, 0.0033704, 0.0009562, 0.0034502, -0.0014400, 0.0012909
8: -0.0118161, -0.0099361, -0.0118782, -0.0099371, -0.0010047, 0.0011208
9: -0.0031525, -0.0029903, -0.0031524, -0.0029849, -0.0000967, 0.0000867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002550, upper bound: 0.0002515
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002550, upper bound: 0.0002552
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0126913, -0.0110924, -0.0128517, -0.0111509, -0.0008060, 0.0010397
1: -0.0065168, -0.0060660, -0.0065620, -0.0060825, -0.0002273, 0.0002931
2: -0.0095226, -0.0061967, -0.0098563, -0.0063184, -0.0016767, 0.0021627
3: 0.0003671, 0.0008073, 0.0003230, 0.0007912, -0.0002219, 0.0002862
4: 0.0107229, 0.0132085, 0.0108138, 0.0134578, -0.0016163, 0.0012531
5: 0.9984854, 0.9991759, 0.9985106, 0.9992452, -0.0004491, 0.0003481
6: 0.0065088, 0.0071357, 0.0065318, 0.0071985, -0.0004076, 0.0003160
7: 0.0009083, 0.0032475, 0.0009939, 0.0034822, -0.0015211, 0.0011793
8: -0.0117204, -0.0098998, -0.0119031, -0.0099664, -0.0009179, 0.0011839
9: -0.0031556, -0.0029986, -0.0031499, -0.0029828, -0.0001021, 0.0000792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B2_A1_A1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002386, upper bound: 0.0002396
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002386, upper bound: 0.0002396
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0127018, -0.0110922, -0.0128820, -0.0111556, -0.0008160, 0.0010454
1: -0.0065198, -0.0060660, -0.0065706, -0.0060838, -0.0002301, 0.0002947
2: -0.0095445, -0.0061961, -0.0099193, -0.0063280, -0.0016975, 0.0021747
3: 0.0003642, 0.0008073, 0.0003146, 0.0007899, -0.0002246, 0.0002878
4: 0.0107225, 0.0132248, 0.0108210, 0.0135049, -0.0016252, 0.0012686
5: 0.9984852, 0.9991804, 0.9985126, 0.9992583, -0.0004515, 0.0003524
6: 0.0065087, 0.0071398, 0.0065336, 0.0072104, -0.0004099, 0.0003199
7: 0.0009079, 0.0032629, 0.0010006, 0.0035265, -0.0015295, 0.0011939
8: -0.0117324, -0.0098995, -0.0119376, -0.0099717, -0.0009292, 0.0011904
9: -0.0031557, -0.0029975, -0.0031494, -0.0029798, -0.0001027, 0.0000802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A1_A1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002634, upper bound: 0.0002201
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002634, upper bound: 0.0002441
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0127277, -0.0111125, -0.0128789, -0.0111553, -0.0008132, 0.0010213
1: -0.0065271, -0.0060717, -0.0065697, -0.0060838, -0.0002293, 0.0002879
2: -0.0095983, -0.0062385, -0.0099130, -0.0063275, -0.0016917, 0.0021244
3: 0.0003571, 0.0008017, 0.0003155, 0.0007900, -0.0002239, 0.0002811
4: 0.0107541, 0.0132650, 0.0108206, 0.0135002, -0.0015877, 0.0012643
5: 0.9984940, 0.9991917, 0.9985125, 0.9992570, -0.0004411, 0.0003512
6: 0.0065167, 0.0071499, 0.0065335, 0.0072092, -0.0004004, 0.0003188
7: 0.0009377, 0.0033007, 0.0010003, 0.0035221, -0.0014942, 0.0011898
8: -0.0117618, -0.0099227, -0.0119341, -0.0099714, -0.0009260, 0.0011629
9: -0.0031537, -0.0029950, -0.0031495, -0.0029801, -0.0001003, 0.0000799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B2_A1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002566, upper bound: 0.0002349
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002566, upper bound: 0.0002349
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0127554, -0.0111218, -0.0128900, -0.0111550, -0.0008242, 0.0010323
1: -0.0065349, -0.0060743, -0.0065728, -0.0060837, -0.0002324, 0.0002910
2: -0.0096560, -0.0062577, -0.0099359, -0.0063268, -0.0017146, 0.0021474
3: 0.0003495, 0.0007992, 0.0003124, 0.0007900, -0.0002269, 0.0002842
4: 0.0107685, 0.0133081, 0.0108201, 0.0135174, -0.0016049, 0.0012814
5: 0.9984980, 0.9992037, 0.9985123, 0.9992618, -0.0004459, 0.0003560
6: 0.0065203, 0.0071608, 0.0065333, 0.0072135, -0.0004047, 0.0003231
7: 0.0009512, 0.0033413, 0.0009998, 0.0035382, -0.0015103, 0.0012059
8: -0.0117934, -0.0099332, -0.0119467, -0.0099710, -0.0009386, 0.0011755
9: -0.0031527, -0.0029923, -0.0031495, -0.0029790, -0.0001014, 0.0000810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002393, upper bound: 0.0002479
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002393, upper bound: 0.0002480
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0127273, -0.0110970, -0.0128615, -0.0111492, -0.0008494, 0.0010446
1: -0.0065270, -0.0060673, -0.0065648, -0.0060820, -0.0002395, 0.0002945
2: -0.0095975, -0.0062062, -0.0098766, -0.0063148, -0.0017670, 0.0021730
3: 0.0003572, 0.0008060, 0.0003203, 0.0007916, -0.0002338, 0.0002876
4: 0.0107300, 0.0132644, 0.0108112, 0.0134731, -0.0016240, 0.0013206
5: 0.9984874, 0.9991915, 0.9985099, 0.9992495, -0.0004512, 0.0003669
6: 0.0065106, 0.0071498, 0.0065311, 0.0072024, -0.0004095, 0.0003330
7: 0.0009150, 0.0033002, 0.0009914, 0.0034965, -0.0015283, 0.0012428
8: -0.0117614, -0.0099050, -0.0119142, -0.0099645, -0.0009673, 0.0011895
9: -0.0031552, -0.0029950, -0.0031501, -0.0029818, -0.0001026, 0.0000835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002353, upper bound: 0.0002472
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002573, upper bound: 0.0002497
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0127273, -0.0110970, -0.0128873, -0.0111551, -0.0008232, 0.0010581
1: -0.0065270, -0.0060673, -0.0065721, -0.0060837, -0.0002321, 0.0002983
2: -0.0095975, -0.0062062, -0.0099304, -0.0063269, -0.0017125, 0.0022010
3: 0.0003572, 0.0008060, 0.0003132, 0.0007900, -0.0002266, 0.0002913
4: 0.0107300, 0.0132644, 0.0108202, 0.0135132, -0.0016449, 0.0012798
5: 0.9984874, 0.9991915, 0.9985124, 0.9992606, -0.0004570, 0.0003556
6: 0.0065106, 0.0071498, 0.0065334, 0.0072125, -0.0004148, 0.0003227
7: 0.0009150, 0.0033002, 0.0009999, 0.0035343, -0.0015480, 0.0012044
8: -0.0117614, -0.0099050, -0.0119436, -0.0099711, -0.0009374, 0.0012048
9: -0.0031552, -0.0029950, -0.0031495, -0.0029793, -0.0001039, 0.0000809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002353, upper bound: 0.0002479
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002573, upper bound: 0.0002498
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127918, -0.0111233, -0.0128615, -0.0111492, -0.0008899, 0.0010002
1: -0.0065452, -0.0060747, -0.0065648, -0.0060820, -0.0002509, 0.0002820
2: -0.0097317, -0.0062608, -0.0098766, -0.0063148, -0.0018511, 0.0020805
3: 0.0003395, 0.0007988, 0.0003203, 0.0007916, -0.0002450, 0.0002753
4: 0.0107708, 0.0133648, 0.0108112, 0.0134731, -0.0015549, 0.0013834
5: 0.9984987, 0.9992194, 0.9985099, 0.9992495, -0.0004320, 0.0003843
6: 0.0065209, 0.0071751, 0.0065311, 0.0072024, -0.0003921, 0.0003489
7: 0.0009534, 0.0033946, 0.0009914, 0.0034965, -0.0014633, 0.0013019
8: -0.0118349, -0.0099349, -0.0119142, -0.0099645, -0.0010133, 0.0011389
9: -0.0031526, -0.0029887, -0.0031501, -0.0029818, -0.0000983, 0.0000874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002550, upper bound: 0.0002426
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002552, upper bound: 0.0002553
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127918, -0.0111233, -0.0128873, -0.0111551, -0.0008651, 0.0010143
1: -0.0065452, -0.0060747, -0.0065721, -0.0060837, -0.0002439, 0.0002860
2: -0.0097317, -0.0062608, -0.0099304, -0.0063269, -0.0017995, 0.0021100
3: 0.0003395, 0.0007988, 0.0003132, 0.0007900, -0.0002381, 0.0002792
4: 0.0107708, 0.0133648, 0.0108202, 0.0135132, -0.0015769, 0.0013449
5: 0.9984987, 0.9992194, 0.9985124, 0.9992606, -0.0004381, 0.0003736
6: 0.0065209, 0.0071751, 0.0065334, 0.0072125, -0.0003977, 0.0003392
7: 0.0009534, 0.0033946, 0.0009999, 0.0035343, -0.0014840, 0.0012657
8: -0.0118349, -0.0099349, -0.0119436, -0.0099711, -0.0009851, 0.0011550
9: -0.0031526, -0.0029887, -0.0031495, -0.0029793, -0.0000996, 0.0000850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002550, upper bound: 0.0002427
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002552, upper bound: 0.0002553
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0128216, -0.0111254, -0.0126650, -0.0110851, -0.0010040, 0.0007951
1: -0.0065535, -0.0060753, -0.0065094, -0.0060640, -0.0002831, 0.0002242
2: -0.0097936, -0.0062652, -0.0094679, -0.0061813, -0.0020885, 0.0016539
3: 0.0003313, 0.0007982, 0.0003744, 0.0008093, -0.0002764, 0.0002189
4: 0.0107741, 0.0134110, 0.0107114, 0.0131676, -0.0012360, 0.0015608
5: 0.9984996, 0.9992322, 0.9984822, 0.9991646, -0.0003434, 0.0004336
6: 0.0065217, 0.0071867, 0.0065059, 0.0071253, -0.0003117, 0.0003936
7: 0.0009565, 0.0034381, 0.0008975, 0.0032090, -0.0011632, 0.0014689
8: -0.0118688, -0.0099373, -0.0116905, -0.0098914, -0.0011432, 0.0009053
9: -0.0031524, -0.0029858, -0.0031564, -0.0030011, -0.0000781, 0.0000986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A1_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002545
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002585
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0128216, -0.0111254, -0.0127277, -0.0111125, -0.0009948, 0.0008873
1: -0.0065535, -0.0060753, -0.0065271, -0.0060717, -0.0002805, 0.0002502
2: -0.0097936, -0.0062652, -0.0095983, -0.0062385, -0.0020693, 0.0018457
3: 0.0003313, 0.0007982, 0.0003571, 0.0008017, -0.0002738, 0.0002442
4: 0.0107741, 0.0134110, 0.0107541, 0.0132650, -0.0013794, 0.0015465
5: 0.9984996, 0.9992322, 0.9984940, 0.9991917, -0.0003832, 0.0004297
6: 0.0065217, 0.0071867, 0.0065167, 0.0071499, -0.0003479, 0.0003900
7: 0.0009565, 0.0034381, 0.0009377, 0.0033007, -0.0012981, 0.0014554
8: -0.0118688, -0.0099373, -0.0117618, -0.0099227, -0.0011328, 0.0010103
9: -0.0031524, -0.0029858, -0.0031537, -0.0029950, -0.0000872, 0.0000977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002546
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002585
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0128327, -0.0111251, -0.0126930, -0.0110927, -0.0010152, 0.0008052
1: -0.0065567, -0.0060752, -0.0065173, -0.0060661, -0.0002862, 0.0002270
2: -0.0098167, -0.0062646, -0.0095261, -0.0061972, -0.0021117, 0.0016750
3: 0.0003282, 0.0007983, 0.0003667, 0.0008072, -0.0002795, 0.0002217
4: 0.0107737, 0.0134283, 0.0107232, 0.0132111, -0.0012518, 0.0015782
5: 0.9984995, 0.9992370, 0.9984855, 0.9991767, -0.0003478, 0.0004385
6: 0.0065216, 0.0071911, 0.0065089, 0.0071363, -0.0003157, 0.0003980
7: 0.0009561, 0.0034543, 0.0009086, 0.0032499, -0.0011780, 0.0014852
8: -0.0118814, -0.0099370, -0.0117223, -0.0099001, -0.0011560, 0.0009169
9: -0.0031524, -0.0029847, -0.0031556, -0.0029984, -0.0000791, 0.0000997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002322
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002600
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0128327, -0.0111251, -0.0127554, -0.0111218, -0.0010040, 0.0008974
1: -0.0065567, -0.0060752, -0.0065349, -0.0060743, -0.0002831, 0.0002530
2: -0.0098167, -0.0062646, -0.0096560, -0.0062577, -0.0020885, 0.0018667
3: 0.0003282, 0.0007983, 0.0003495, 0.0007992, -0.0002764, 0.0002470
4: 0.0107737, 0.0134283, 0.0107685, 0.0133081, -0.0013951, 0.0015609
5: 0.9984995, 0.9992370, 0.9984980, 0.9992037, -0.0003876, 0.0004337
6: 0.0065216, 0.0071911, 0.0065203, 0.0071608, -0.0003518, 0.0003936
7: 0.0009561, 0.0034543, 0.0009512, 0.0033413, -0.0013129, 0.0014689
8: -0.0118814, -0.0099370, -0.0117934, -0.0099332, -0.0011433, 0.0010218
9: -0.0031524, -0.0029847, -0.0031527, -0.0029923, -0.0000882, 0.0000986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002322
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002600
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0127926, -0.0111215, -0.0127478, -0.0111159, -0.0009656, 0.0008984
1: -0.0065454, -0.0060742, -0.0065327, -0.0060727, -0.0002722, 0.0002533
2: -0.0097334, -0.0062571, -0.0096401, -0.0062456, -0.0020086, 0.0018689
3: 0.0003392, 0.0007993, 0.0003516, 0.0008008, -0.0002658, 0.0002473
4: 0.0107680, 0.0133660, 0.0107594, 0.0132963, -0.0013967, 0.0015011
5: 0.9984980, 0.9992197, 0.9984955, 0.9992003, -0.0003881, 0.0004171
6: 0.0065202, 0.0071754, 0.0065180, 0.0071578, -0.0003522, 0.0003786
7: 0.0009508, 0.0033958, 0.0009427, 0.0033301, -0.0013145, 0.0014127
8: -0.0118358, -0.0099329, -0.0117847, -0.0099266, -0.0010995, 0.0010231
9: -0.0031528, -0.0029886, -0.0031533, -0.0029930, -0.0000883, 0.0000949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002530
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002530
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0128037, -0.0111212, -0.0127753, -0.0111243, -0.0009744, 0.0009107
1: -0.0065485, -0.0060741, -0.0065405, -0.0060750, -0.0002747, 0.0002568
2: -0.0097564, -0.0062565, -0.0096974, -0.0062630, -0.0020269, 0.0018944
3: 0.0003362, 0.0007993, 0.0003440, 0.0007985, -0.0002682, 0.0002507
4: 0.0107676, 0.0133832, 0.0107724, 0.0133391, -0.0014157, 0.0015148
5: 0.9984978, 0.9992245, 0.9984992, 0.9992123, -0.0003933, 0.0004209
6: 0.0065201, 0.0071797, 0.0065213, 0.0071686, -0.0003570, 0.0003820
7: 0.0009504, 0.0034119, 0.0009549, 0.0033704, -0.0013324, 0.0014256
8: -0.0118484, -0.0099326, -0.0118161, -0.0099361, -0.0011095, 0.0010370
9: -0.0031528, -0.0029875, -0.0031525, -0.0029903, -0.0000895, 0.0000957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002514, upper bound: 0.0002550
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002514, upper bound: 0.0002549
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0128187, -0.0111255, -0.0127478, -0.0111159, -0.0009752, 0.0008698
1: -0.0065527, -0.0060753, -0.0065327, -0.0060727, -0.0002750, 0.0002452
2: -0.0097876, -0.0062654, -0.0096401, -0.0062456, -0.0020287, 0.0018094
3: 0.0003321, 0.0007982, 0.0003516, 0.0008008, -0.0002685, 0.0002394
4: 0.0107742, 0.0134065, 0.0107594, 0.0132963, -0.0013522, 0.0015161
5: 0.9984996, 0.9992309, 0.9984955, 0.9992003, -0.0003757, 0.0004212
6: 0.0065218, 0.0071856, 0.0065180, 0.0071578, -0.0003410, 0.0003823
7: 0.0009566, 0.0034339, 0.0009427, 0.0033301, -0.0012726, 0.0014268
8: -0.0118654, -0.0099374, -0.0117847, -0.0099266, -0.0011105, 0.0009905
9: -0.0031524, -0.0029860, -0.0031533, -0.0029930, -0.0000855, 0.0000958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002590
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002590
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0128298, -0.0111252, -0.0127753, -0.0111243, -0.0009842, 0.0008824
1: -0.0065559, -0.0060753, -0.0065405, -0.0060750, -0.0002775, 0.0002488
2: -0.0098108, -0.0062648, -0.0096974, -0.0062630, -0.0020474, 0.0018355
3: 0.0003290, 0.0007983, 0.0003440, 0.0007985, -0.0002709, 0.0002429
4: 0.0107738, 0.0134239, 0.0107724, 0.0133391, -0.0013717, 0.0015301
5: 0.9984995, 0.9992357, 0.9984992, 0.9992123, -0.0003811, 0.0004251
6: 0.0065217, 0.0071900, 0.0065213, 0.0071686, -0.0003459, 0.0003859
7: 0.0009562, 0.0034502, 0.0009549, 0.0033704, -0.0012909, 0.0014400
8: -0.0118782, -0.0099371, -0.0118161, -0.0099361, -0.0011208, 0.0010047
9: -0.0031524, -0.0029849, -0.0031525, -0.0029903, -0.0000867, 0.0000967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002446, upper bound: 0.0002601
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002446, upper bound: 0.0002601
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0128517, -0.0111509, -0.0126913, -0.0110924, -0.0010397, 0.0008060
1: -0.0065620, -0.0060825, -0.0065168, -0.0060660, -0.0002931, 0.0002273
2: -0.0098563, -0.0063184, -0.0095226, -0.0061967, -0.0021627, 0.0016767
3: 0.0003230, 0.0007912, 0.0003671, 0.0008073, -0.0002862, 0.0002219
4: 0.0108138, 0.0134578, 0.0107229, 0.0132085, -0.0012531, 0.0016163
5: 0.9985106, 0.9992452, 0.9984854, 0.9991759, -0.0003481, 0.0004491
6: 0.0065318, 0.0071985, 0.0065088, 0.0071357, -0.0003160, 0.0004076
7: 0.0009939, 0.0034822, 0.0009083, 0.0032475, -0.0011793, 0.0015211
8: -0.0119031, -0.0099664, -0.0117204, -0.0098998, -0.0011839, 0.0009179
9: -0.0031499, -0.0029828, -0.0031556, -0.0029986, -0.0000792, 0.0001021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002396, upper bound: 0.0002386
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002396, upper bound: 0.0002443
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128820, -0.0111556, -0.0127018, -0.0110922, -0.0010454, 0.0008160
1: -0.0065706, -0.0060838, -0.0065198, -0.0060660, -0.0002947, 0.0002301
2: -0.0099193, -0.0063280, -0.0095445, -0.0061961, -0.0021747, 0.0016975
3: 0.0003146, 0.0007899, 0.0003642, 0.0008073, -0.0002878, 0.0002246
4: 0.0108210, 0.0135049, 0.0107225, 0.0132248, -0.0012686, 0.0016252
5: 0.9985126, 0.9992583, 0.9984852, 0.9991804, -0.0003524, 0.0004515
6: 0.0065336, 0.0072104, 0.0065087, 0.0071398, -0.0003199, 0.0004099
7: 0.0010006, 0.0035265, 0.0009079, 0.0032629, -0.0011939, 0.0015295
8: -0.0119376, -0.0099717, -0.0117324, -0.0098995, -0.0011904, 0.0009292
9: -0.0031494, -0.0029798, -0.0031557, -0.0029975, -0.0000802, 0.0001027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002635
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002637
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0128789, -0.0111553, -0.0127277, -0.0111125, -0.0010213, 0.0008132
1: -0.0065697, -0.0060838, -0.0065271, -0.0060717, -0.0002879, 0.0002293
2: -0.0099130, -0.0063275, -0.0095983, -0.0062385, -0.0021244, 0.0016917
3: 0.0003155, 0.0007900, 0.0003571, 0.0008017, -0.0002811, 0.0002239
4: 0.0108206, 0.0135002, 0.0107541, 0.0132650, -0.0012643, 0.0015877
5: 0.9985125, 0.9992570, 0.9984940, 0.9991917, -0.0003512, 0.0004411
6: 0.0065335, 0.0072092, 0.0065167, 0.0071499, -0.0003188, 0.0004004
7: 0.0010003, 0.0035221, 0.0009377, 0.0033007, -0.0011898, 0.0014942
8: -0.0119341, -0.0099714, -0.0117618, -0.0099227, -0.0011629, 0.0009260
9: -0.0031495, -0.0029801, -0.0031537, -0.0029950, -0.0000799, 0.0001003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002306, upper bound: 0.0002603
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002306, upper bound: 0.0002637
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0128900, -0.0111550, -0.0127554, -0.0111218, -0.0010323, 0.0008242
1: -0.0065728, -0.0060837, -0.0065349, -0.0060743, -0.0002910, 0.0002324
2: -0.0099359, -0.0063268, -0.0096560, -0.0062577, -0.0021474, 0.0017146
3: 0.0003124, 0.0007900, 0.0003495, 0.0007992, -0.0002842, 0.0002269
4: 0.0108201, 0.0135174, 0.0107685, 0.0133081, -0.0012814, 0.0016049
5: 0.9985123, 0.9992618, 0.9984980, 0.9992037, -0.0003560, 0.0004459
6: 0.0065333, 0.0072135, 0.0065203, 0.0071608, -0.0003231, 0.0004047
7: 0.0009998, 0.0035382, 0.0009512, 0.0033413, -0.0012059, 0.0015103
8: -0.0119467, -0.0099710, -0.0117934, -0.0099332, -0.0011755, 0.0009386
9: -0.0031495, -0.0029790, -0.0031527, -0.0029923, -0.0000810, 0.0001014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002442, upper bound: 0.0002450
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002442, upper bound: 0.0002638
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0128615, -0.0111492, -0.0127273, -0.0110970, -0.0010446, 0.0008494
1: -0.0065648, -0.0060820, -0.0065270, -0.0060673, -0.0002945, 0.0002395
2: -0.0098766, -0.0063148, -0.0095975, -0.0062062, -0.0021730, 0.0017670
3: 0.0003203, 0.0007916, 0.0003572, 0.0008060, -0.0002876, 0.0002338
4: 0.0108112, 0.0134731, 0.0107300, 0.0132644, -0.0013206, 0.0016240
5: 0.9985099, 0.9992495, 0.9984874, 0.9991915, -0.0003669, 0.0004512
6: 0.0065311, 0.0072024, 0.0065106, 0.0071498, -0.0003330, 0.0004095
7: 0.0009914, 0.0034965, 0.0009150, 0.0033002, -0.0012428, 0.0015283
8: -0.0119142, -0.0099645, -0.0117614, -0.0099050, -0.0011895, 0.0009673
9: -0.0031501, -0.0029818, -0.0031552, -0.0029950, -0.0000835, 0.0001026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002396, upper bound: 0.0002353
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002441, upper bound: 0.0002573
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128873, -0.0111551, -0.0127273, -0.0110970, -0.0010581, 0.0008232
1: -0.0065721, -0.0060837, -0.0065270, -0.0060673, -0.0002983, 0.0002321
2: -0.0099304, -0.0063269, -0.0095975, -0.0062062, -0.0022010, 0.0017125
3: 0.0003132, 0.0007900, 0.0003572, 0.0008060, -0.0002913, 0.0002266
4: 0.0108202, 0.0135132, 0.0107300, 0.0132644, -0.0012798, 0.0016449
5: 0.9985124, 0.9992606, 0.9984874, 0.9991915, -0.0003556, 0.0004570
6: 0.0065334, 0.0072125, 0.0065106, 0.0071498, -0.0003227, 0.0004148
7: 0.0009999, 0.0035343, 0.0009150, 0.0033002, -0.0012044, 0.0015480
8: -0.0119436, -0.0099711, -0.0117614, -0.0099050, -0.0012048, 0.0009374
9: -0.0031495, -0.0029793, -0.0031552, -0.0029950, -0.0000809, 0.0001039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002396, upper bound: 0.0002444
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002441, upper bound: 0.0002637
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0128615, -0.0111492, -0.0127918, -0.0111233, -0.0010002, 0.0008899
1: -0.0065648, -0.0060820, -0.0065452, -0.0060747, -0.0002820, 0.0002509
2: -0.0098766, -0.0063148, -0.0097317, -0.0062608, -0.0020805, 0.0018511
3: 0.0003203, 0.0007916, 0.0003395, 0.0007988, -0.0002753, 0.0002450
4: 0.0108112, 0.0134731, 0.0107708, 0.0133648, -0.0013834, 0.0015549
5: 0.9985099, 0.9992495, 0.9984987, 0.9992194, -0.0003843, 0.0004320
6: 0.0065311, 0.0072024, 0.0065209, 0.0071751, -0.0003489, 0.0003921
7: 0.0009914, 0.0034965, 0.0009534, 0.0033946, -0.0013019, 0.0014633
8: -0.0119142, -0.0099645, -0.0118349, -0.0099349, -0.0011389, 0.0010133
9: -0.0031501, -0.0029818, -0.0031526, -0.0029887, -0.0000874, 0.0000983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002305, upper bound: 0.0002574
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002445, upper bound: 0.0002576
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128873, -0.0111551, -0.0127918, -0.0111233, -0.0010143, 0.0008651
1: -0.0065721, -0.0060837, -0.0065452, -0.0060747, -0.0002860, 0.0002439
2: -0.0099304, -0.0063269, -0.0097317, -0.0062608, -0.0021100, 0.0017995
3: 0.0003132, 0.0007900, 0.0003395, 0.0007988, -0.0002792, 0.0002381
4: 0.0108202, 0.0135132, 0.0107708, 0.0133648, -0.0013449, 0.0015769
5: 0.9985124, 0.9992606, 0.9984987, 0.9992194, -0.0003736, 0.0004381
6: 0.0065334, 0.0072125, 0.0065209, 0.0071751, -0.0003392, 0.0003977
7: 0.0009999, 0.0035343, 0.0009534, 0.0033946, -0.0012657, 0.0014840
8: -0.0119436, -0.0099711, -0.0118349, -0.0099349, -0.0011550, 0.0009851
9: -0.0031495, -0.0029793, -0.0031526, -0.0029887, -0.0000850, 0.0000996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002305, upper bound: 0.0002637
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002445, upper bound: 0.0002641
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0127926, -0.0111215, -0.0127937, -0.0111211, -0.0007631, 0.0007615
1: -0.0065454, -0.0060742, -0.0065457, -0.0060741, -0.0002152, 0.0002147
2: -0.0097334, -0.0062571, -0.0097357, -0.0062563, -0.0015875, 0.0015840
3: 0.0003392, 0.0007993, 0.0003389, 0.0007994, -0.0002101, 0.0002096
4: 0.0107680, 0.0133660, 0.0107674, 0.0133677, -0.0011838, 0.0011864
5: 0.9984980, 0.9992197, 0.9984977, 0.9992202, -0.0003289, 0.0003296
6: 0.0065202, 0.0071754, 0.0065201, 0.0071758, -0.0002985, 0.0002992
7: 0.0009508, 0.0033958, 0.0009502, 0.0033974, -0.0011141, 0.0011165
8: -0.0118358, -0.0099329, -0.0118370, -0.0099324, -0.0008690, 0.0008671
9: -0.0031528, -0.0029886, -0.0031528, -0.0029885, -0.0000748, 0.0000750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A1_A1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002257, upper bound: 0.0002537
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_B1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002257, upper bound: 0.0002536
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0127926, -0.0111215, -0.0128517, -0.0111509, -0.0007520, 0.0008547
1: -0.0065454, -0.0060742, -0.0065620, -0.0060825, -0.0002120, 0.0002410
2: -0.0097334, -0.0062571, -0.0098563, -0.0063184, -0.0015643, 0.0017780
3: 0.0003392, 0.0007993, 0.0003230, 0.0007912, -0.0002070, 0.0002353
4: 0.0107680, 0.0133660, 0.0108138, 0.0134578, -0.0013288, 0.0011690
5: 0.9984980, 0.9992197, 0.9985106, 0.9992452, -0.0003692, 0.0003248
6: 0.0065202, 0.0071754, 0.0065318, 0.0071985, -0.0003351, 0.0002948
7: 0.0009508, 0.0033958, 0.0009939, 0.0034822, -0.0012505, 0.0011002
8: -0.0118358, -0.0099329, -0.0119031, -0.0099664, -0.0008563, 0.0009733
9: -0.0031528, -0.0029886, -0.0031499, -0.0029828, -0.0000840, 0.0000739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A1_A1_B1_B2_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002257, upper bound: 0.0002537
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_B2_B2

### Relational analysis result of IS_A2_B2_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002257, upper bound: 0.0002537
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0128037, -0.0111212, -0.0128246, -0.0111256, -0.0007751, 0.0007688
1: -0.0065485, -0.0060741, -0.0065544, -0.0060754, -0.0002185, 0.0002168
2: -0.0097564, -0.0062565, -0.0097999, -0.0062656, -0.0016125, 0.0015993
3: 0.0003362, 0.0007993, 0.0003304, 0.0007981, -0.0002134, 0.0002116
4: 0.0107676, 0.0133832, 0.0107744, 0.0134157, -0.0011952, 0.0012051
5: 0.9984978, 0.9992245, 0.9984996, 0.9992335, -0.0003321, 0.0003348
6: 0.0065201, 0.0071797, 0.0065218, 0.0071879, -0.0003014, 0.0003039
7: 0.0009504, 0.0034119, 0.0009567, 0.0034425, -0.0011248, 0.0011341
8: -0.0118484, -0.0099326, -0.0118722, -0.0099375, -0.0008827, 0.0008755
9: -0.0031528, -0.0029875, -0.0031524, -0.0029855, -0.0000755, 0.0000762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002295
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002552
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0128037, -0.0111212, -0.0128820, -0.0111556, -0.0007639, 0.0008621
1: -0.0065485, -0.0060741, -0.0065706, -0.0060838, -0.0002154, 0.0002430
2: -0.0097564, -0.0062565, -0.0099193, -0.0063280, -0.0015891, 0.0017933
3: 0.0003362, 0.0007993, 0.0003146, 0.0007899, -0.0002103, 0.0002373
4: 0.0107676, 0.0133832, 0.0108210, 0.0135049, -0.0013402, 0.0011876
5: 0.9984978, 0.9992245, 0.9985126, 0.9992583, -0.0003723, 0.0003300
6: 0.0065201, 0.0071797, 0.0065336, 0.0072104, -0.0003380, 0.0002995
7: 0.0009504, 0.0034119, 0.0010006, 0.0035265, -0.0012613, 0.0011177
8: -0.0118484, -0.0099326, -0.0119376, -0.0099717, -0.0008699, 0.0009816
9: -0.0031528, -0.0029875, -0.0031494, -0.0029798, -0.0000847, 0.0000751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002295
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002295
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0128187, -0.0111255, -0.0128186, -0.0111461, -0.0007837, 0.0007944
1: -0.0065527, -0.0060753, -0.0065527, -0.0060812, -0.0002209, 0.0002240
2: -0.0097876, -0.0062654, -0.0097875, -0.0063084, -0.0016302, 0.0016526
3: 0.0003321, 0.0007982, 0.0003321, 0.0007925, -0.0002157, 0.0002187
4: 0.0107742, 0.0134065, 0.0108064, 0.0134064, -0.0012350, 0.0012183
5: 0.9984996, 0.9992309, 0.9985085, 0.9992310, -0.0003431, 0.0003385
6: 0.0065218, 0.0071856, 0.0065299, 0.0071856, -0.0003115, 0.0003072
7: 0.0009566, 0.0034339, 0.0009868, 0.0034338, -0.0011623, 0.0011465
8: -0.0118654, -0.0099374, -0.0118654, -0.0099609, -0.0008924, 0.0009046
9: -0.0031524, -0.0029860, -0.0031504, -0.0029860, -0.0000780, 0.0000770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002252, upper bound: 0.0002592
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002252, upper bound: 0.0002592
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0128298, -0.0111252, -0.0128491, -0.0111503, -0.0007952, 0.0008028
1: -0.0065559, -0.0060753, -0.0065613, -0.0060824, -0.0002242, 0.0002264
2: -0.0098108, -0.0062648, -0.0098509, -0.0063171, -0.0016542, 0.0016701
3: 0.0003290, 0.0007983, 0.0003237, 0.0007913, -0.0002189, 0.0002210
4: 0.0107738, 0.0134239, 0.0108129, 0.0134538, -0.0012481, 0.0012362
5: 0.9984995, 0.9992357, 0.9985103, 0.9992442, -0.0003468, 0.0003435
6: 0.0065217, 0.0071900, 0.0065315, 0.0071975, -0.0003148, 0.0003118
7: 0.0009562, 0.0034502, 0.0009930, 0.0034784, -0.0011746, 0.0011634
8: -0.0118782, -0.0099371, -0.0119001, -0.0099657, -0.0009055, 0.0009142
9: -0.0031524, -0.0029849, -0.0031499, -0.0029831, -0.0000789, 0.0000781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002447, upper bound: 0.0002602
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002447, upper bound: 0.0002603
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0128187, -0.0111255, -0.0128448, -0.0111515, -0.0007563, 0.0008065
1: -0.0065527, -0.0060753, -0.0065601, -0.0060827, -0.0002132, 0.0002274
2: -0.0097876, -0.0062654, -0.0098420, -0.0063196, -0.0015733, 0.0016776
3: 0.0003321, 0.0007982, 0.0003249, 0.0007910, -0.0002082, 0.0002220
4: 0.0107742, 0.0134065, 0.0108147, 0.0134472, -0.0012537, 0.0011758
5: 0.9984996, 0.9992309, 0.9985109, 0.9992422, -0.0003483, 0.0003267
6: 0.0065218, 0.0071856, 0.0065320, 0.0071958, -0.0003162, 0.0002965
7: 0.0009566, 0.0034339, 0.0009947, 0.0034721, -0.0011799, 0.0011066
8: -0.0118654, -0.0099374, -0.0118952, -0.0099671, -0.0008612, 0.0009183
9: -0.0031524, -0.0029860, -0.0031498, -0.0029835, -0.0000792, 0.0000743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002267, upper bound: 0.0002593
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002267, upper bound: 0.0002593
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0128298, -0.0111252, -0.0128755, -0.0111561, -0.0007687, 0.0008138
1: -0.0065559, -0.0060753, -0.0065687, -0.0060840, -0.0002167, 0.0002294
2: -0.0098108, -0.0062648, -0.0099057, -0.0063292, -0.0015990, 0.0016929
3: 0.0003290, 0.0007983, 0.0003164, 0.0007897, -0.0002116, 0.0002240
4: 0.0107738, 0.0134239, 0.0108219, 0.0134948, -0.0012651, 0.0011950
5: 0.9984995, 0.9992357, 0.9985129, 0.9992555, -0.0003515, 0.0003320
6: 0.0065217, 0.0071900, 0.0065338, 0.0072079, -0.0003191, 0.0003014
7: 0.0009562, 0.0034502, 0.0010015, 0.0035169, -0.0011906, 0.0011246
8: -0.0118782, -0.0099371, -0.0119301, -0.0099723, -0.0008753, 0.0009267
9: -0.0031524, -0.0029849, -0.0031494, -0.0029805, -0.0000799, 0.0000755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002448, upper bound: 0.0002602
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002448, upper bound: 0.0002602
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0128233, -0.0111456, -0.0128216, -0.0111254, -0.0008240, 0.0007778
1: -0.0065540, -0.0060810, -0.0065535, -0.0060753, -0.0002323, 0.0002193
2: -0.0097972, -0.0063073, -0.0097936, -0.0062652, -0.0017141, 0.0016180
3: 0.0003308, 0.0007926, 0.0003313, 0.0007982, -0.0002268, 0.0002141
4: 0.0108055, 0.0134137, 0.0107741, 0.0134110, -0.0012092, 0.0012810
5: 0.9985083, 0.9992330, 0.9984996, 0.9992322, -0.0003359, 0.0003559
6: 0.0065297, 0.0071874, 0.0065217, 0.0071867, -0.0003049, 0.0003230
7: 0.0009861, 0.0034406, 0.0009565, 0.0034381, -0.0011380, 0.0012056
8: -0.0118707, -0.0099603, -0.0118688, -0.0099373, -0.0009383, 0.0008857
9: -0.0031504, -0.0029856, -0.0031524, -0.0029858, -0.0000764, 0.0000810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002450, upper bound: 0.0002365
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002450, upper bound: 0.0002365
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128534, -0.0111498, -0.0128327, -0.0111251, -0.0008325, 0.0007903
1: -0.0065625, -0.0060822, -0.0065567, -0.0060752, -0.0002347, 0.0002228
2: -0.0098598, -0.0063160, -0.0098167, -0.0062646, -0.0017319, 0.0016439
3: 0.0003225, 0.0007915, 0.0003282, 0.0007983, -0.0002292, 0.0002175
4: 0.0108121, 0.0134605, 0.0107737, 0.0134283, -0.0012286, 0.0012943
5: 0.9985102, 0.9992460, 0.9984995, 0.9992370, -0.0003413, 0.0003596
6: 0.0065313, 0.0071992, 0.0065216, 0.0071911, -0.0003098, 0.0003264
7: 0.0009922, 0.0034847, 0.0009561, 0.0034543, -0.0011562, 0.0012181
8: -0.0119050, -0.0099651, -0.0118814, -0.0099370, -0.0009480, 0.0008999
9: -0.0031500, -0.0029826, -0.0031524, -0.0029847, -0.0000776, 0.0000818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002294, upper bound: 0.0002571
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002294, upper bound: 0.0002575
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0128505, -0.0111496, -0.0128517, -0.0111509, -0.0007791, 0.0007781
1: -0.0065617, -0.0060821, -0.0065620, -0.0060825, -0.0002197, 0.0002194
2: -0.0098539, -0.0063155, -0.0098563, -0.0063184, -0.0016207, 0.0016186
3: 0.0003233, 0.0007915, 0.0003230, 0.0007912, -0.0002145, 0.0002142
4: 0.0108117, 0.0134560, 0.0108138, 0.0134578, -0.0012097, 0.0012112
5: 0.9985101, 0.9992447, 0.9985106, 0.9992452, -0.0003361, 0.0003365
6: 0.0065312, 0.0071981, 0.0065318, 0.0071985, -0.0003051, 0.0003054
7: 0.0009919, 0.0034805, 0.0009939, 0.0034822, -0.0011384, 0.0011399
8: -0.0119017, -0.0099648, -0.0119031, -0.0099664, -0.0008871, 0.0008860
9: -0.0031500, -0.0029829, -0.0031499, -0.0029828, -0.0000764, 0.0000765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002323, upper bound: 0.0002574
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002323, upper bound: 0.0002571
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0128615, -0.0111492, -0.0128820, -0.0111556, -0.0007912, 0.0007851
1: -0.0065648, -0.0060820, -0.0065706, -0.0060838, -0.0002231, 0.0002213
2: -0.0098766, -0.0063148, -0.0099193, -0.0063280, -0.0016459, 0.0016331
3: 0.0003203, 0.0007916, 0.0003146, 0.0007899, -0.0002178, 0.0002161
4: 0.0108112, 0.0134731, 0.0108210, 0.0135049, -0.0012205, 0.0012300
5: 0.9985099, 0.9992495, 0.9985126, 0.9992583, -0.0003391, 0.0003417
6: 0.0065311, 0.0072024, 0.0065336, 0.0072104, -0.0003078, 0.0003102
7: 0.0009914, 0.0034965, 0.0010006, 0.0035265, -0.0011486, 0.0011576
8: -0.0119142, -0.0099645, -0.0119376, -0.0099717, -0.0009010, 0.0008940
9: -0.0031501, -0.0029818, -0.0031494, -0.0029798, -0.0000771, 0.0000777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002500, upper bound: 0.0002382
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B2_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002500, upper bound: 0.0002577
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0128873, -0.0111551, -0.0128037, -0.0111212, -0.0009007, 0.0007602
1: -0.0065721, -0.0060837, -0.0065485, -0.0060741, -0.0002539, 0.0002143
2: -0.0099304, -0.0063269, -0.0097564, -0.0062565, -0.0018736, 0.0015814
3: 0.0003132, 0.0007900, 0.0003362, 0.0007993, -0.0002479, 0.0002093
4: 0.0108202, 0.0135132, 0.0107676, 0.0133832, -0.0011818, 0.0014002
5: 0.9985124, 0.9992606, 0.9984978, 0.9992245, -0.0003283, 0.0003890
6: 0.0065334, 0.0072125, 0.0065201, 0.0071797, -0.0002980, 0.0003531
7: 0.0009999, 0.0035343, 0.0009504, 0.0034119, -0.0011122, 0.0013178
8: -0.0119436, -0.0099711, -0.0118484, -0.0099326, -0.0010256, 0.0008657
9: -0.0031495, -0.0029793, -0.0031528, -0.0029875, -0.0000747, 0.0000885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002425, upper bound: 0.0002445
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002444, upper bound: 0.0002638
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0128873, -0.0111551, -0.0128298, -0.0111252, -0.0008727, 0.0007692
1: -0.0065721, -0.0060837, -0.0065559, -0.0060753, -0.0002461, 0.0002169
2: -0.0099304, -0.0063269, -0.0098108, -0.0062648, -0.0018154, 0.0016001
3: 0.0003132, 0.0007900, 0.0003290, 0.0007983, -0.0002402, 0.0002117
4: 0.0108202, 0.0135132, 0.0107738, 0.0134239, -0.0011958, 0.0013567
5: 0.9985124, 0.9992606, 0.9984995, 0.9992357, -0.0003322, 0.0003769
6: 0.0065334, 0.0072125, 0.0065217, 0.0071900, -0.0003016, 0.0003421
7: 0.0009999, 0.0035343, 0.0009562, 0.0034502, -0.0011254, 0.0012768
8: -0.0119436, -0.0099711, -0.0118782, -0.0099371, -0.0009938, 0.0008759
9: -0.0031495, -0.0029793, -0.0031524, -0.0029849, -0.0000756, 0.0000857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002425, upper bound: 0.0002445
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002444, upper bound: 0.0002638
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0128873, -0.0111551, -0.0128615, -0.0111492, -0.0008249, 0.0007872
1: -0.0065721, -0.0060837, -0.0065648, -0.0060820, -0.0002326, 0.0002219
2: -0.0099304, -0.0063269, -0.0098766, -0.0063148, -0.0017161, 0.0016376
3: 0.0003132, 0.0007900, 0.0003203, 0.0007916, -0.0002271, 0.0002167
4: 0.0108202, 0.0135132, 0.0108112, 0.0134731, -0.0012238, 0.0012825
5: 0.9985124, 0.9992606, 0.9985099, 0.9992495, -0.0003400, 0.0003563
6: 0.0065334, 0.0072125, 0.0065311, 0.0072024, -0.0003086, 0.0003234
7: 0.0009999, 0.0035343, 0.0009914, 0.0034965, -0.0011518, 0.0012070
8: -0.0119436, -0.0099711, -0.0119142, -0.0099645, -0.0009394, 0.0008964
9: -0.0031495, -0.0029793, -0.0031501, -0.0029818, -0.0000773, 0.0000810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002442, upper bound: 0.0002450
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002444, upper bound: 0.0002639
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0128873, -0.0111551, -0.0128873, -0.0111551, -0.0008010, 0.0008010
1: -0.0065721, -0.0060837, -0.0065721, -0.0060837, -0.0002258, 0.0002258
2: -0.0099304, -0.0063269, -0.0099304, -0.0063269, -0.0016662, 0.0016662
3: 0.0003132, 0.0007900, 0.0003132, 0.0007900, -0.0002205, 0.0002205
4: 0.0108202, 0.0135132, 0.0108202, 0.0135132, -0.0012452, 0.0012452
5: 0.9985124, 0.9992606, 0.9985124, 0.9992606, -0.0003460, 0.0003460
6: 0.0065334, 0.0072125, 0.0065334, 0.0072125, -0.0003140, 0.0003140
7: 0.0009999, 0.0035343, 0.0009999, 0.0035343, -0.0011719, 0.0011719
8: -0.0119436, -0.0099711, -0.0119436, -0.0099711, -0.0009121, 0.0009121
9: -0.0031495, -0.0029793, -0.0031495, -0.0029793, -0.0000787, 0.0000787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002442, upper bound: 0.0002452
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002444, upper bound: 0.0002641
time: 0.64 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.30 seconds
IS_A1_B1_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002205, upper bound: 0.0002397
IS_A1_B1_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002205, upper bound: 0.0002397
IS_A1_B1_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002205, upper bound: 0.0002397
IS_A1_B1_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002205, upper bound: 0.0002397
IS_A1_B1_A1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002201
IS_A1_B1_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002447
IS_A1_B1_A1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002201
IS_A1_B1_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002448
IS_A1_B1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002472
IS_A1_B1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002472
IS_A1_B1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002449, upper bound: 0.0002501
IS_A1_B1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002449, upper bound: 0.0002502
IS_A1_B1_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002480
IS_A1_B1_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002480
IS_A1_B1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002449, upper bound: 0.0002502
IS_A1_B1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002449, upper bound: 0.0002502
IS_A1_B1_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002421, upper bound: 0.0002303
IS_A1_B1_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002421, upper bound: 0.0002303
IS_A1_B1_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002473
IS_A1_B1_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002478
IS_A1_B1_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002327, upper bound: 0.0002478
IS_A1_B1_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002327, upper bound: 0.0002479
IS_A1_B1_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002503, upper bound: 0.0002349
IS_A1_B1_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002503, upper bound: 0.0002480
IS_A1_B1_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002397, upper bound: 0.0002413
IS_A1_B1_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002447, upper bound: 0.0002552
IS_A1_B1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002397, upper bound: 0.0002418
IS_A1_B1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002447, upper bound: 0.0002552
IS_A1_B1_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002448, upper bound: 0.0002426
IS_A1_B1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002450, upper bound: 0.0002553
IS_A1_B1_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002448, upper bound: 0.0002428
IS_A1_B1_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002450, upper bound: 0.0002556
IS_A1_B2_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002545, upper bound: 0.0002201
IS_A1_B2_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002545, upper bound: 0.0002201
IS_A1_B2_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002545, upper bound: 0.0002303
IS_A1_B2_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002545, upper bound: 0.0002303
IS_A1_B2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002321, upper bound: 0.0002396
IS_A1_B2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002321, upper bound: 0.0002445
IS_A1_B2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002321, upper bound: 0.0002472
IS_A1_B2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002321, upper bound: 0.0002477
IS_A1_B2_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002530, upper bound: 0.0002255
IS_A1_B2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002530, upper bound: 0.0002413
IS_A1_B2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002550, upper bound: 0.0002514
IS_A1_B2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002550, upper bound: 0.0002551
IS_A1_B2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002542, upper bound: 0.0002275
IS_A1_B2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002542, upper bound: 0.0002417
IS_A1_B2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002550, upper bound: 0.0002515
IS_A1_B2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002550, upper bound: 0.0002552
IS_A1_B2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002386, upper bound: 0.0002396
IS_A1_B2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002386, upper bound: 0.0002396
IS_A1_B2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002634, upper bound: 0.0002201
IS_A1_B2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002634, upper bound: 0.0002441
IS_A1_B2_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002566, upper bound: 0.0002349
IS_A1_B2_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002566, upper bound: 0.0002349
IS_A1_B2_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002393, upper bound: 0.0002479
IS_A1_B2_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002393, upper bound: 0.0002480
IS_A1_B2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002353, upper bound: 0.0002472
IS_A1_B2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002573, upper bound: 0.0002497
IS_A1_B2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002353, upper bound: 0.0002479
IS_A1_B2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002573, upper bound: 0.0002498
IS_A1_B2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002550, upper bound: 0.0002426
IS_A1_B2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002552, upper bound: 0.0002553
IS_A1_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002550, upper bound: 0.0002427
IS_A1_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002552, upper bound: 0.0002553
IS_A2_B1_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002545
IS_A2_B1_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002585
IS_A2_B1_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002546
IS_A2_B1_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002585
IS_A2_B1_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002322
IS_A2_B1_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002600
IS_A2_B1_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002322
IS_A2_B1_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002600
IS_A2_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002530
IS_A2_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002255, upper bound: 0.0002530
IS_A2_B1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002514, upper bound: 0.0002550
IS_A2_B1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002514, upper bound: 0.0002549
IS_A2_B1_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002590
IS_A2_B1_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002590
IS_A2_B1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002446, upper bound: 0.0002601
IS_A2_B1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002446, upper bound: 0.0002601
IS_A2_B1_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002396, upper bound: 0.0002386
IS_A2_B1_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002396, upper bound: 0.0002443
IS_A2_B1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002635
IS_A2_B1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002637
IS_A2_B1_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002306, upper bound: 0.0002603
IS_A2_B1_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002306, upper bound: 0.0002637
IS_A2_B1_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002442, upper bound: 0.0002450
IS_A2_B1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002442, upper bound: 0.0002638
IS_A2_B1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002396, upper bound: 0.0002353
IS_A2_B1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002441, upper bound: 0.0002573
IS_A2_B1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002396, upper bound: 0.0002444
IS_A2_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002441, upper bound: 0.0002637
IS_A2_B1_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002305, upper bound: 0.0002574
IS_A2_B1_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002445, upper bound: 0.0002576
IS_A2_B1_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002305, upper bound: 0.0002637
IS_A2_B1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002445, upper bound: 0.0002641
IS_A2_B2_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002257, upper bound: 0.0002537
IS_A2_B2_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002257, upper bound: 0.0002536
IS_A2_B2_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002257, upper bound: 0.0002537
IS_A2_B2_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002257, upper bound: 0.0002537
IS_A2_B2_A1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002295
IS_A2_B2_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002552
IS_A2_B2_A1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002295
IS_A2_B2_A1_A1_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002295
IS_A2_B2_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002252, upper bound: 0.0002592
IS_A2_B2_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002252, upper bound: 0.0002592
IS_A2_B2_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002447, upper bound: 0.0002602
IS_A2_B2_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002447, upper bound: 0.0002603
IS_A2_B2_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002267, upper bound: 0.0002593
IS_A2_B2_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002267, upper bound: 0.0002593
IS_A2_B2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002448, upper bound: 0.0002602
IS_A2_B2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002448, upper bound: 0.0002602
IS_A2_B2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002450, upper bound: 0.0002365
IS_A2_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002450, upper bound: 0.0002365
IS_A2_B2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002294, upper bound: 0.0002571
IS_A2_B2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002294, upper bound: 0.0002575
IS_A2_B2_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002323, upper bound: 0.0002574
IS_A2_B2_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002323, upper bound: 0.0002571
IS_A2_B2_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002500, upper bound: 0.0002382
IS_A2_B2_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002500, upper bound: 0.0002577
IS_A2_B2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002425, upper bound: 0.0002445
IS_A2_B2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002444, upper bound: 0.0002638
IS_A2_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002425, upper bound: 0.0002445
IS_A2_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002444, upper bound: 0.0002638
IS_A2_B2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002442, upper bound: 0.0002450
IS_A2_B2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002444, upper bound: 0.0002639
IS_A2_B2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002442, upper bound: 0.0002452
IS_A2_B2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 5, lower bound: -0.0002444, upper bound: 0.0002641

## BFS IS instance: IS_A1_B1_A1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0126913, -0.0110924, -0.0126650, -0.0110851, -0.0007417, 0.0007124
1: -0.0065168, -0.0060660, -0.0065094, -0.0060640, -0.0002091, 0.0002009
2: -0.0095226, -0.0061967, -0.0094679, -0.0061813, -0.0015428, 0.0014820
3: 0.0003671, 0.0008073, 0.0003744, 0.0008093, -0.0002042, 0.0001961
4: 0.0107229, 0.0132085, 0.0107114, 0.0131676, -0.0011075, 0.0011530
5: 0.9984854, 0.9991759, 0.9984822, 0.9991646, -0.0003077, 0.0003203
6: 0.0065088, 0.0071357, 0.0065059, 0.0071253, -0.0002793, 0.0002908
7: 0.0009083, 0.0032475, 0.0008975, 0.0032090, -0.0010423, 0.0010851
8: -0.0117204, -0.0098998, -0.0116905, -0.0098914, -0.0008446, 0.0008112
9: -0.0031556, -0.0029986, -0.0031564, -0.0030011, -0.0000700, 0.0000729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002204, upper bound: 0.0002233
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002204, upper bound: 0.0002397
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0126913, -0.0110924, -0.0126912, -0.0110904, -0.0007392, 0.0007483
1: -0.0065168, -0.0060660, -0.0065168, -0.0060654, -0.0002084, 0.0002110
2: -0.0095226, -0.0061967, -0.0095225, -0.0061924, -0.0015376, 0.0015567
3: 0.0003671, 0.0008073, 0.0003671, 0.0008078, -0.0002035, 0.0002060
4: 0.0107229, 0.0132085, 0.0107197, 0.0132084, -0.0011634, 0.0011491
5: 0.9984854, 0.9991759, 0.9984844, 0.9991760, -0.0003232, 0.0003193
6: 0.0065088, 0.0071357, 0.0065080, 0.0071356, -0.0002934, 0.0002898
7: 0.0009083, 0.0032475, 0.0009053, 0.0032474, -0.0010949, 0.0010815
8: -0.0117204, -0.0098998, -0.0117204, -0.0098974, -0.0008417, 0.0008521
9: -0.0031556, -0.0029986, -0.0031558, -0.0029986, -0.0000735, 0.0000726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002204, upper bound: 0.0002233
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002204, upper bound: 0.0002397
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0126913, -0.0110924, -0.0127277, -0.0111125, -0.0007325, 0.0008046
1: -0.0065168, -0.0060660, -0.0065271, -0.0060717, -0.0002065, 0.0002269
2: -0.0095226, -0.0061967, -0.0095983, -0.0062385, -0.0015237, 0.0016738
3: 0.0003671, 0.0008073, 0.0003571, 0.0008017, -0.0002016, 0.0002215
4: 0.0107229, 0.0132085, 0.0107541, 0.0132650, -0.0012509, 0.0011387
5: 0.9984854, 0.9991759, 0.9984940, 0.9991917, -0.0003475, 0.0003164
6: 0.0065088, 0.0071357, 0.0065167, 0.0071499, -0.0003155, 0.0002872
7: 0.0009083, 0.0032475, 0.0009377, 0.0033007, -0.0011772, 0.0010717
8: -0.0117204, -0.0098998, -0.0117618, -0.0099227, -0.0008341, 0.0009162
9: -0.0031556, -0.0029986, -0.0031537, -0.0029950, -0.0000790, 0.0000720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002341, upper bound: 0.0002236
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002341, upper bound: 0.0002397
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0126913, -0.0110924, -0.0127550, -0.0111155, -0.0007307, 0.0008405
1: -0.0065168, -0.0060660, -0.0065348, -0.0060725, -0.0002060, 0.0002370
2: -0.0095226, -0.0061967, -0.0096551, -0.0062447, -0.0015201, 0.0017484
3: 0.0003671, 0.0008073, 0.0003496, 0.0008009, -0.0002012, 0.0002314
4: 0.0107229, 0.0132085, 0.0107588, 0.0133075, -0.0013067, 0.0011360
5: 0.9984854, 0.9991759, 0.9984954, 0.9992034, -0.0003630, 0.0003156
6: 0.0065088, 0.0071357, 0.0065179, 0.0071606, -0.0003295, 0.0002865
7: 0.0009083, 0.0032475, 0.0009421, 0.0033407, -0.0012297, 0.0010691
8: -0.0117204, -0.0098998, -0.0117929, -0.0099261, -0.0008321, 0.0009571
9: -0.0031556, -0.0029986, -0.0031534, -0.0029923, -0.0000826, 0.0000718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002341, upper bound: 0.0002236
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002341, upper bound: 0.0002397
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0126930, -0.0110927, -0.0127211, -0.0110974, -0.0007197, 0.0007462
1: -0.0065173, -0.0060661, -0.0065252, -0.0060674, -0.0002029, 0.0002104
2: -0.0095261, -0.0061972, -0.0095846, -0.0062070, -0.0014972, 0.0015523
3: 0.0003667, 0.0008072, 0.0003589, 0.0008059, -0.0001981, 0.0002054
4: 0.0107232, 0.0132111, 0.0107306, 0.0132548, -0.0011601, 0.0011189
5: 0.9984855, 0.9991767, 0.9984875, 0.9991888, -0.0003223, 0.0003109
6: 0.0065089, 0.0071363, 0.0065108, 0.0071473, -0.0002926, 0.0002822
7: 0.0009086, 0.0032499, 0.0009156, 0.0032911, -0.0010918, 0.0010530
8: -0.0117223, -0.0099001, -0.0117544, -0.0099055, -0.0008196, 0.0008497
9: -0.0031556, -0.0029984, -0.0031551, -0.0029956, -0.0000733, 0.0000707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002204, upper bound: 0.0002449
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002204, upper bound: 0.0002449
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0126930, -0.0110927, -0.0127847, -0.0111238, -0.0007121, 0.0008404
1: -0.0065173, -0.0060661, -0.0065431, -0.0060749, -0.0002008, 0.0002369
2: -0.0095261, -0.0061972, -0.0097170, -0.0062619, -0.0014814, 0.0017482
3: 0.0003667, 0.0008072, 0.0003414, 0.0007986, -0.0001960, 0.0002313
4: 0.0107232, 0.0132111, 0.0107716, 0.0133537, -0.0013065, 0.0011071
5: 0.9984855, 0.9991767, 0.9984989, 0.9992163, -0.0003630, 0.0003076
6: 0.0065089, 0.0071363, 0.0065211, 0.0071723, -0.0003295, 0.0002792
7: 0.0009086, 0.0032499, 0.0009542, 0.0033842, -0.0012295, 0.0010419
8: -0.0117223, -0.0099001, -0.0118268, -0.0099355, -0.0008109, 0.0009570
9: -0.0031556, -0.0029984, -0.0031525, -0.0029894, -0.0000826, 0.0000700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002341, upper bound: 0.0002447
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002341, upper bound: 0.0002447
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0127168, -0.0110973, -0.0126650, -0.0110851, -0.0007759, 0.0007080
1: -0.0065240, -0.0060674, -0.0065094, -0.0060640, -0.0002187, 0.0001996
2: -0.0095757, -0.0062067, -0.0094679, -0.0061813, -0.0016140, 0.0014727
3: 0.0003601, 0.0008059, 0.0003744, 0.0008093, -0.0002136, 0.0001949
4: 0.0107304, 0.0132482, 0.0107114, 0.0131676, -0.0011006, 0.0012062
5: 0.9984875, 0.9991870, 0.9984822, 0.9991646, -0.0003058, 0.0003351
6: 0.0065107, 0.0071457, 0.0065059, 0.0071253, -0.0002776, 0.0003042
7: 0.0009153, 0.0032849, 0.0008975, 0.0032090, -0.0010358, 0.0011352
8: -0.0117495, -0.0099053, -0.0116905, -0.0098914, -0.0008835, 0.0008062
9: -0.0031552, -0.0029960, -0.0031564, -0.0030011, -0.0000696, 0.0000762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002313
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002472
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0127168, -0.0110973, -0.0127277, -0.0111125, -0.0007667, 0.0008002
1: -0.0065240, -0.0060674, -0.0065271, -0.0060717, -0.0002162, 0.0002256
2: -0.0095757, -0.0062067, -0.0095983, -0.0062385, -0.0015949, 0.0016645
3: 0.0003601, 0.0008059, 0.0003571, 0.0008017, -0.0002111, 0.0002203
4: 0.0107304, 0.0132482, 0.0107541, 0.0132650, -0.0012439, 0.0011919
5: 0.9984875, 0.9991870, 0.9984940, 0.9991917, -0.0003456, 0.0003311
6: 0.0065107, 0.0071457, 0.0065167, 0.0071499, -0.0003137, 0.0003006
7: 0.0009153, 0.0032849, 0.0009377, 0.0033007, -0.0011707, 0.0011217
8: -0.0117495, -0.0099053, -0.0117618, -0.0099227, -0.0008730, 0.0009111
9: -0.0031552, -0.0029960, -0.0031537, -0.0029950, -0.0000786, 0.0000753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002313
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002472
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0127273, -0.0110970, -0.0126930, -0.0110927, -0.0007881, 0.0007157
1: -0.0065270, -0.0060673, -0.0065173, -0.0060661, -0.0002222, 0.0002018
2: -0.0095975, -0.0062062, -0.0095261, -0.0061972, -0.0016393, 0.0014887
3: 0.0003572, 0.0008060, 0.0003667, 0.0008072, -0.0002169, 0.0001970
4: 0.0107300, 0.0132644, 0.0107232, 0.0132111, -0.0011126, 0.0012251
5: 0.9984874, 0.9991915, 0.9984855, 0.9991767, -0.0003091, 0.0003404
6: 0.0065106, 0.0071498, 0.0065089, 0.0071363, -0.0002806, 0.0003090
7: 0.0009150, 0.0033002, 0.0009086, 0.0032499, -0.0010471, 0.0011530
8: -0.0117614, -0.0099050, -0.0117223, -0.0099001, -0.0008974, 0.0008149
9: -0.0031552, -0.0029950, -0.0031556, -0.0029984, -0.0000703, 0.0000774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002255
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002502
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0127273, -0.0110970, -0.0127554, -0.0111218, -0.0007769, 0.0008078
1: -0.0065270, -0.0060673, -0.0065349, -0.0060743, -0.0002190, 0.0002278
2: -0.0095975, -0.0062062, -0.0096560, -0.0062577, -0.0016161, 0.0016805
3: 0.0003572, 0.0008060, 0.0003495, 0.0007992, -0.0002139, 0.0002224
4: 0.0107300, 0.0132644, 0.0107685, 0.0133081, -0.0012559, 0.0012078
5: 0.9984874, 0.9991915, 0.9984980, 0.9992037, -0.0003489, 0.0003356
6: 0.0065106, 0.0071498, 0.0065203, 0.0071608, -0.0003167, 0.0003046
7: 0.0009150, 0.0033002, 0.0009512, 0.0033413, -0.0011819, 0.0011367
8: -0.0117614, -0.0099050, -0.0117934, -0.0099332, -0.0008847, 0.0009199
9: -0.0031552, -0.0029950, -0.0031527, -0.0029923, -0.0000794, 0.0000763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002255
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002502
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0127168, -0.0110973, -0.0126912, -0.0110904, -0.0007499, 0.0007207
1: -0.0065240, -0.0060674, -0.0065168, -0.0060654, -0.0002114, 0.0002032
2: -0.0095757, -0.0062067, -0.0095225, -0.0061924, -0.0015600, 0.0014992
3: 0.0003601, 0.0008059, 0.0003671, 0.0008078, -0.0002064, 0.0001984
4: 0.0107304, 0.0132482, 0.0107197, 0.0132084, -0.0011204, 0.0011658
5: 0.9984875, 0.9991870, 0.9984844, 0.9991760, -0.0003113, 0.0003239
6: 0.0065107, 0.0071457, 0.0065080, 0.0071356, -0.0002826, 0.0002940
7: 0.0009153, 0.0032849, 0.0009053, 0.0032474, -0.0010544, 0.0010972
8: -0.0117495, -0.0099053, -0.0117204, -0.0098974, -0.0008539, 0.0008207
9: -0.0031552, -0.0029960, -0.0031558, -0.0029986, -0.0000708, 0.0000737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002329
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002480
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0127168, -0.0110973, -0.0127550, -0.0111155, -0.0007394, 0.0008123
1: -0.0065240, -0.0060674, -0.0065348, -0.0060725, -0.0002085, 0.0002290
2: -0.0095757, -0.0062067, -0.0096551, -0.0062447, -0.0015381, 0.0016898
3: 0.0003601, 0.0008059, 0.0003496, 0.0008009, -0.0002035, 0.0002236
4: 0.0107304, 0.0132482, 0.0107588, 0.0133075, -0.0012629, 0.0011495
5: 0.9984875, 0.9991870, 0.9984954, 0.9992034, -0.0003509, 0.0003194
6: 0.0065107, 0.0071457, 0.0065179, 0.0071606, -0.0003185, 0.0002899
7: 0.0009153, 0.0032849, 0.0009421, 0.0033407, -0.0011885, 0.0010818
8: -0.0117495, -0.0099053, -0.0117929, -0.0099261, -0.0008420, 0.0009250
9: -0.0031552, -0.0029960, -0.0031534, -0.0029923, -0.0000798, 0.0000726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002329
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002480
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0127273, -0.0110970, -0.0127188, -0.0110975, -0.0007630, 0.0007263
1: -0.0065270, -0.0060673, -0.0065246, -0.0060675, -0.0002151, 0.0002048
2: -0.0095975, -0.0062062, -0.0095798, -0.0062072, -0.0015872, 0.0015109
3: 0.0003572, 0.0008060, 0.0003596, 0.0008059, -0.0002100, 0.0001999
4: 0.0107300, 0.0132644, 0.0107307, 0.0132512, -0.0011292, 0.0011862
5: 0.9984874, 0.9991915, 0.9984875, 0.9991878, -0.0003137, 0.0003296
6: 0.0065106, 0.0071498, 0.0065108, 0.0071464, -0.0002848, 0.0002991
7: 0.0009150, 0.0033002, 0.0009157, 0.0032877, -0.0010627, 0.0011163
8: -0.0117614, -0.0099050, -0.0117517, -0.0099055, -0.0008688, 0.0008271
9: -0.0031552, -0.0029950, -0.0031551, -0.0029959, -0.0000714, 0.0000750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002275
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002501
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0127273, -0.0110970, -0.0127830, -0.0111239, -0.0007504, 0.0008179
1: -0.0065270, -0.0060673, -0.0065427, -0.0060749, -0.0002116, 0.0002306
2: -0.0095975, -0.0062062, -0.0097134, -0.0062621, -0.0015610, 0.0017015
3: 0.0003572, 0.0008060, 0.0003419, 0.0007986, -0.0002066, 0.0002252
4: 0.0107300, 0.0132644, 0.0107717, 0.0133511, -0.0012716, 0.0011666
5: 0.9984874, 0.9991915, 0.9984990, 0.9992156, -0.0003533, 0.0003241
6: 0.0065106, 0.0071498, 0.0065211, 0.0071716, -0.0003207, 0.0002942
7: 0.0009150, 0.0033002, 0.0009543, 0.0033817, -0.0011967, 0.0010979
8: -0.0117614, -0.0099050, -0.0118249, -0.0099356, -0.0008545, 0.0009314
9: -0.0031552, -0.0029950, -0.0031525, -0.0029895, -0.0000804, 0.0000737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002275
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002500
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0127277, -0.0111125, -0.0126913, -0.0110924, -0.0008046, 0.0007325
1: -0.0065271, -0.0060717, -0.0065168, -0.0060660, -0.0002269, 0.0002065
2: -0.0095983, -0.0062385, -0.0095226, -0.0061967, -0.0016738, 0.0015237
3: 0.0003571, 0.0008017, 0.0003671, 0.0008073, -0.0002215, 0.0002016
4: 0.0107541, 0.0132650, 0.0107229, 0.0132085, -0.0011387, 0.0012509
5: 0.9984940, 0.9991917, 0.9984854, 0.9991759, -0.0003164, 0.0003475
6: 0.0065167, 0.0071499, 0.0065088, 0.0071357, -0.0002872, 0.0003155
7: 0.0009377, 0.0033007, 0.0009083, 0.0032475, -0.0010717, 0.0011772
8: -0.0117618, -0.0099227, -0.0117204, -0.0098998, -0.0009162, 0.0008341
9: -0.0031537, -0.0029950, -0.0031556, -0.0029986, -0.0000720, 0.0000790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002241, upper bound: 0.0002303
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002241, upper bound: 0.0002303
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0127277, -0.0111125, -0.0127168, -0.0110973, -0.0008002, 0.0007667
1: -0.0065271, -0.0060717, -0.0065240, -0.0060674, -0.0002256, 0.0002162
2: -0.0095983, -0.0062385, -0.0095757, -0.0062067, -0.0016645, 0.0015949
3: 0.0003571, 0.0008017, 0.0003601, 0.0008059, -0.0002203, 0.0002111
4: 0.0107541, 0.0132650, 0.0107304, 0.0132482, -0.0011919, 0.0012439
5: 0.9984940, 0.9991917, 0.9984875, 0.9991870, -0.0003311, 0.0003456
6: 0.0065167, 0.0071499, 0.0065107, 0.0071457, -0.0003006, 0.0003137
7: 0.0009377, 0.0033007, 0.0009153, 0.0032849, -0.0011217, 0.0011707
8: -0.0117618, -0.0099227, -0.0117495, -0.0099053, -0.0009111, 0.0008730
9: -0.0031537, -0.0029950, -0.0031552, -0.0029960, -0.0000753, 0.0000786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002241, upper bound: 0.0002302
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002241, upper bound: 0.0002303
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127554, -0.0111218, -0.0126935, -0.0110903, -0.0008398, 0.0007295
1: -0.0065349, -0.0060743, -0.0065174, -0.0060654, -0.0002368, 0.0002057
2: -0.0096560, -0.0062577, -0.0095272, -0.0061922, -0.0017469, 0.0015175
3: 0.0003495, 0.0007992, 0.0003665, 0.0008079, -0.0002312, 0.0002008
4: 0.0107685, 0.0133081, 0.0107195, 0.0132119, -0.0011341, 0.0013055
5: 0.9984980, 0.9992037, 0.9984844, 0.9991769, -0.0003151, 0.0003627
6: 0.0065203, 0.0071608, 0.0065080, 0.0071365, -0.0002860, 0.0003292
7: 0.0009512, 0.0033413, 0.0009052, 0.0032507, -0.0010673, 0.0012286
8: -0.0117934, -0.0099332, -0.0117229, -0.0098973, -0.0009562, 0.0008307
9: -0.0031527, -0.0029923, -0.0031558, -0.0029983, -0.0000717, 0.0000825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002205, upper bound: 0.0002473
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002205, upper bound: 0.0002473
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127554, -0.0111218, -0.0127211, -0.0110974, -0.0008119, 0.0007377
1: -0.0065349, -0.0060743, -0.0065252, -0.0060674, -0.0002289, 0.0002080
2: -0.0096560, -0.0062577, -0.0095846, -0.0062070, -0.0016890, 0.0015345
3: 0.0003495, 0.0007992, 0.0003589, 0.0008059, -0.0002235, 0.0002031
4: 0.0107685, 0.0133081, 0.0107306, 0.0132548, -0.0011468, 0.0012622
5: 0.9984980, 0.9992037, 0.9984875, 0.9991888, -0.0003186, 0.0003507
6: 0.0065203, 0.0071608, 0.0065108, 0.0071473, -0.0002892, 0.0003183
7: 0.0009512, 0.0033413, 0.0009156, 0.0032911, -0.0010792, 0.0011879
8: -0.0117934, -0.0099332, -0.0117544, -0.0099055, -0.0009245, 0.0008400
9: -0.0031527, -0.0029923, -0.0031551, -0.0029956, -0.0000725, 0.0000798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002205, upper bound: 0.0002478
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002205, upper bound: 0.0002478
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0127536, -0.0111215, -0.0127277, -0.0111125, -0.0007598, 0.0007303
1: -0.0065344, -0.0060742, -0.0065271, -0.0060717, -0.0002142, 0.0002059
2: -0.0096523, -0.0062571, -0.0095983, -0.0062385, -0.0015805, 0.0015193
3: 0.0003500, 0.0007993, 0.0003571, 0.0008017, -0.0002092, 0.0002011
4: 0.0107681, 0.0133054, 0.0107541, 0.0132650, -0.0011354, 0.0011812
5: 0.9984980, 0.9992029, 0.9984940, 0.9991917, -0.0003155, 0.0003282
6: 0.0065202, 0.0071601, 0.0065167, 0.0071499, -0.0002863, 0.0002979
7: 0.0009508, 0.0033387, 0.0009377, 0.0033007, -0.0010685, 0.0011116
8: -0.0117914, -0.0099329, -0.0117618, -0.0099227, -0.0008652, 0.0008317
9: -0.0031528, -0.0029924, -0.0031537, -0.0029950, -0.0000718, 0.0000746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002326, upper bound: 0.0002438
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002326, upper bound: 0.0002478
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0127536, -0.0111215, -0.0127550, -0.0111155, -0.0007570, 0.0007660
1: -0.0065344, -0.0060742, -0.0065348, -0.0060725, -0.0002134, 0.0002160
2: -0.0096523, -0.0062571, -0.0096551, -0.0062447, -0.0015748, 0.0015934
3: 0.0003500, 0.0007993, 0.0003496, 0.0008009, -0.0002084, 0.0002109
4: 0.0107681, 0.0133054, 0.0107588, 0.0133075, -0.0011908, 0.0011769
5: 0.9984980, 0.9992029, 0.9984954, 0.9992034, -0.0003308, 0.0003270
6: 0.0065202, 0.0071601, 0.0065179, 0.0071606, -0.0003003, 0.0002968
7: 0.0009508, 0.0033387, 0.0009421, 0.0033407, -0.0011207, 0.0011076
8: -0.0117914, -0.0099329, -0.0117929, -0.0099261, -0.0008620, 0.0008722
9: -0.0031528, -0.0029924, -0.0031534, -0.0029923, -0.0000753, 0.0000744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002326, upper bound: 0.0002439
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002326, upper bound: 0.0002478
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0127277, -0.0111125, -0.0127847, -0.0111238, -0.0007301, 0.0007956
1: -0.0065271, -0.0060717, -0.0065431, -0.0060749, -0.0002058, 0.0002243
2: -0.0095983, -0.0062385, -0.0097170, -0.0062619, -0.0015187, 0.0016549
3: 0.0003571, 0.0008017, 0.0003414, 0.0007986, -0.0002010, 0.0002190
4: 0.0107541, 0.0132650, 0.0107716, 0.0133537, -0.0012368, 0.0011350
5: 0.9984940, 0.9991917, 0.9984989, 0.9992163, -0.0003436, 0.0003153
6: 0.0065167, 0.0071499, 0.0065211, 0.0071723, -0.0003119, 0.0002862
7: 0.0009377, 0.0033007, 0.0009542, 0.0033842, -0.0011639, 0.0010682
8: -0.0117618, -0.0099227, -0.0118268, -0.0099355, -0.0008314, 0.0009059
9: -0.0031537, -0.0029950, -0.0031525, -0.0029894, -0.0000782, 0.0000717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002326, upper bound: 0.0002349
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002326, upper bound: 0.0002349
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127554, -0.0111218, -0.0127847, -0.0111238, -0.0007386, 0.0007654
1: -0.0065349, -0.0060743, -0.0065431, -0.0060749, -0.0002082, 0.0002158
2: -0.0096560, -0.0062577, -0.0097170, -0.0062619, -0.0015365, 0.0015921
3: 0.0003495, 0.0007992, 0.0003414, 0.0007986, -0.0002033, 0.0002107
4: 0.0107685, 0.0133081, 0.0107716, 0.0133537, -0.0011898, 0.0011483
5: 0.9984980, 0.9992037, 0.9984989, 0.9992163, -0.0003306, 0.0003190
6: 0.0065203, 0.0071608, 0.0065211, 0.0071723, -0.0003001, 0.0002896
7: 0.0009512, 0.0033413, 0.0009542, 0.0033842, -0.0011198, 0.0010807
8: -0.0117934, -0.0099332, -0.0118268, -0.0099355, -0.0008411, 0.0008715
9: -0.0031527, -0.0029923, -0.0031525, -0.0029894, -0.0000752, 0.0000726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002326, upper bound: 0.0002480
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002326, upper bound: 0.0002481
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0127550, -0.0111155, -0.0126913, -0.0110924, -0.0008405, 0.0007307
1: -0.0065348, -0.0060725, -0.0065168, -0.0060660, -0.0002370, 0.0002060
2: -0.0096551, -0.0062447, -0.0095226, -0.0061967, -0.0017484, 0.0015201
3: 0.0003496, 0.0008009, 0.0003671, 0.0008073, -0.0002314, 0.0002012
4: 0.0107588, 0.0133075, 0.0107229, 0.0132085, -0.0011360, 0.0013067
5: 0.9984954, 0.9992034, 0.9984854, 0.9991759, -0.0003156, 0.0003630
6: 0.0065179, 0.0071606, 0.0065088, 0.0071357, -0.0002865, 0.0003295
7: 0.0009421, 0.0033407, 0.0009083, 0.0032475, -0.0010691, 0.0012297
8: -0.0117929, -0.0099261, -0.0117204, -0.0098998, -0.0009571, 0.0008321
9: -0.0031534, -0.0029923, -0.0031556, -0.0029986, -0.0000718, 0.0000826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002413
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002413
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127830, -0.0111239, -0.0127018, -0.0110922, -0.0008463, 0.0007415
1: -0.0065427, -0.0060749, -0.0065198, -0.0060660, -0.0002386, 0.0002090
2: -0.0097134, -0.0062621, -0.0095445, -0.0061961, -0.0017605, 0.0015424
3: 0.0003419, 0.0007986, 0.0003642, 0.0008073, -0.0002330, 0.0002041
4: 0.0107717, 0.0133511, 0.0107225, 0.0132248, -0.0011527, 0.0013157
5: 0.9984990, 0.9992156, 0.9984852, 0.9991804, -0.0003203, 0.0003655
6: 0.0065211, 0.0071716, 0.0065087, 0.0071398, -0.0002907, 0.0003318
7: 0.0009543, 0.0033817, 0.0009079, 0.0032629, -0.0010848, 0.0012382
8: -0.0118249, -0.0099356, -0.0117324, -0.0098995, -0.0009637, 0.0008443
9: -0.0031525, -0.0029895, -0.0031557, -0.0029975, -0.0000728, 0.0000831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002549
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002201, upper bound: 0.0002552
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0127550, -0.0111155, -0.0127168, -0.0110973, -0.0008123, 0.0007394
1: -0.0065348, -0.0060725, -0.0065240, -0.0060674, -0.0002290, 0.0002085
2: -0.0096551, -0.0062447, -0.0095757, -0.0062067, -0.0016898, 0.0015381
3: 0.0003496, 0.0008009, 0.0003601, 0.0008059, -0.0002236, 0.0002035
4: 0.0107588, 0.0133075, 0.0107304, 0.0132482, -0.0011495, 0.0012629
5: 0.9984954, 0.9992034, 0.9984875, 0.9991870, -0.0003194, 0.0003509
6: 0.0065179, 0.0071606, 0.0065107, 0.0071457, -0.0002899, 0.0003185
7: 0.0009421, 0.0033407, 0.0009153, 0.0032849, -0.0010818, 0.0011885
8: -0.0117929, -0.0099261, -0.0117495, -0.0099053, -0.0009250, 0.0008420
9: -0.0031534, -0.0029923, -0.0031552, -0.0029960, -0.0000726, 0.0000798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002418
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002418
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127830, -0.0111239, -0.0127273, -0.0110970, -0.0008179, 0.0007504
1: -0.0065427, -0.0060749, -0.0065270, -0.0060673, -0.0002306, 0.0002116
2: -0.0097134, -0.0062621, -0.0095975, -0.0062062, -0.0017015, 0.0015610
3: 0.0003419, 0.0007986, 0.0003572, 0.0008060, -0.0002252, 0.0002066
4: 0.0107717, 0.0133511, 0.0107300, 0.0132644, -0.0011666, 0.0012716
5: 0.9984990, 0.9992156, 0.9984874, 0.9991915, -0.0003241, 0.0003533
6: 0.0065211, 0.0071716, 0.0065106, 0.0071498, -0.0002942, 0.0003207
7: 0.0009543, 0.0033817, 0.0009150, 0.0033002, -0.0010979, 0.0011967
8: -0.0118249, -0.0099356, -0.0117614, -0.0099050, -0.0009314, 0.0008545
9: -0.0031525, -0.0029895, -0.0031552, -0.0029950, -0.0000737, 0.0000804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002549
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002234, upper bound: 0.0002551
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0127550, -0.0111155, -0.0127536, -0.0111215, -0.0007660, 0.0007570
1: -0.0065348, -0.0060725, -0.0065344, -0.0060742, -0.0002160, 0.0002134
2: -0.0096551, -0.0062447, -0.0096523, -0.0062571, -0.0015934, 0.0015748
3: 0.0003496, 0.0008009, 0.0003500, 0.0007993, -0.0002109, 0.0002084
4: 0.0107588, 0.0133075, 0.0107681, 0.0133054, -0.0011769, 0.0011908
5: 0.9984954, 0.9992034, 0.9984980, 0.9992029, -0.0003270, 0.0003308
6: 0.0065179, 0.0071606, 0.0065202, 0.0071601, -0.0002968, 0.0003003
7: 0.0009421, 0.0033407, 0.0009508, 0.0033387, -0.0011076, 0.0011207
8: -0.0117929, -0.0099261, -0.0117914, -0.0099329, -0.0008722, 0.0008620
9: -0.0031534, -0.0029923, -0.0031528, -0.0029924, -0.0000744, 0.0000753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002306, upper bound: 0.0002426
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002306, upper bound: 0.0002427
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127830, -0.0111239, -0.0127641, -0.0111211, -0.0007729, 0.0007680
1: -0.0065427, -0.0060749, -0.0065373, -0.0060741, -0.0002179, 0.0002165
2: -0.0097134, -0.0062621, -0.0096741, -0.0062564, -0.0016079, 0.0015975
3: 0.0003419, 0.0007986, 0.0003471, 0.0007994, -0.0002128, 0.0002114
4: 0.0107717, 0.0133511, 0.0107675, 0.0133217, -0.0011939, 0.0012016
5: 0.9984990, 0.9992156, 0.9984977, 0.9992074, -0.0003317, 0.0003338
6: 0.0065211, 0.0071716, 0.0065201, 0.0071642, -0.0003011, 0.0003030
7: 0.0009543, 0.0033817, 0.0009503, 0.0033541, -0.0011236, 0.0011309
8: -0.0118249, -0.0099356, -0.0118033, -0.0099325, -0.0008802, 0.0008745
9: -0.0031525, -0.0029895, -0.0031528, -0.0029914, -0.0000754, 0.0000759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002306, upper bound: 0.0002551
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002306, upper bound: 0.0002553
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0127550, -0.0111155, -0.0127813, -0.0111236, -0.0007390, 0.0007685
1: -0.0065348, -0.0060725, -0.0065422, -0.0060748, -0.0002083, 0.0002167
2: -0.0096551, -0.0062447, -0.0097098, -0.0062615, -0.0015372, 0.0015986
3: 0.0003496, 0.0008009, 0.0003424, 0.0007987, -0.0002034, 0.0002115
4: 0.0107588, 0.0133075, 0.0107713, 0.0133483, -0.0011947, 0.0011488
5: 0.9984954, 0.9992034, 0.9984989, 0.9992148, -0.0003319, 0.0003192
6: 0.0065179, 0.0071606, 0.0065210, 0.0071709, -0.0003013, 0.0002897
7: 0.0009421, 0.0033407, 0.0009539, 0.0033791, -0.0011243, 0.0010812
8: -0.0117929, -0.0099261, -0.0118229, -0.0099353, -0.0008415, 0.0008751
9: -0.0031534, -0.0029923, -0.0031526, -0.0029897, -0.0000755, 0.0000726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002309, upper bound: 0.0002428
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002309, upper bound: 0.0002428
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127830, -0.0111239, -0.0127918, -0.0111233, -0.0007454, 0.0007811
1: -0.0065427, -0.0060749, -0.0065452, -0.0060747, -0.0002102, 0.0002202
2: -0.0097134, -0.0062621, -0.0097317, -0.0062608, -0.0015506, 0.0016249
3: 0.0003419, 0.0007986, 0.0003395, 0.0007988, -0.0002052, 0.0002150
4: 0.0107717, 0.0133511, 0.0107708, 0.0133648, -0.0012143, 0.0011588
5: 0.9984990, 0.9992156, 0.9984987, 0.9992194, -0.0003374, 0.0003220
6: 0.0065211, 0.0071716, 0.0065209, 0.0071751, -0.0003062, 0.0002922
7: 0.0009543, 0.0033817, 0.0009534, 0.0033946, -0.0011428, 0.0010906
8: -0.0118249, -0.0099356, -0.0118349, -0.0099349, -0.0008488, 0.0008895
9: -0.0031525, -0.0029895, -0.0031526, -0.0029887, -0.0000767, 0.0000732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002308, upper bound: 0.0002553
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002308, upper bound: 0.0002556
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0126650, -0.0110851, -0.0127926, -0.0111215, -0.0007960, 0.0009761
1: -0.0065094, -0.0060640, -0.0065454, -0.0060742, -0.0002244, 0.0002752
2: -0.0094679, -0.0061813, -0.0097334, -0.0062571, -0.0016558, 0.0020304
3: 0.0003744, 0.0008093, 0.0003392, 0.0007993, -0.0002191, 0.0002687
4: 0.0107114, 0.0131676, 0.0107680, 0.0133660, -0.0015174, 0.0012374
5: 0.9984822, 0.9991646, 0.9984980, 0.9992197, -0.0004216, 0.0003438
6: 0.0065059, 0.0071253, 0.0065202, 0.0071754, -0.0003827, 0.0003121
7: 0.0008975, 0.0032090, 0.0009508, 0.0033958, -0.0014280, 0.0011646
8: -0.0116905, -0.0098914, -0.0118358, -0.0099329, -0.0009064, 0.0011114
9: -0.0031564, -0.0030011, -0.0031528, -0.0029886, -0.0000959, 0.0000782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002335, upper bound: 0.0002201
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002335, upper bound: 0.0002201
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0126650, -0.0110851, -0.0128187, -0.0111255, -0.0007913, 0.0010003
1: -0.0065094, -0.0060640, -0.0065527, -0.0060753, -0.0002231, 0.0002820
2: -0.0094679, -0.0061813, -0.0097876, -0.0062654, -0.0016461, 0.0020809
3: 0.0003744, 0.0008093, 0.0003321, 0.0007982, -0.0002178, 0.0002754
4: 0.0107114, 0.0131676, 0.0107742, 0.0134065, -0.0015551, 0.0012302
5: 0.9984822, 0.9991646, 0.9984996, 0.9992309, -0.0004321, 0.0003418
6: 0.0065059, 0.0071253, 0.0065218, 0.0071856, -0.0003922, 0.0003102
7: 0.0008975, 0.0032090, 0.0009566, 0.0034339, -0.0014635, 0.0011578
8: -0.0116905, -0.0098914, -0.0118654, -0.0099374, -0.0009011, 0.0011391
9: -0.0031564, -0.0030011, -0.0031524, -0.0029860, -0.0000983, 0.0000777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002335, upper bound: 0.0002201
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002335, upper bound: 0.0002201
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127277, -0.0111125, -0.0127926, -0.0111215, -0.0008882, 0.0009669
1: -0.0065271, -0.0060717, -0.0065454, -0.0060742, -0.0002504, 0.0002726
2: -0.0095983, -0.0062385, -0.0097334, -0.0062571, -0.0018476, 0.0020113
3: 0.0003571, 0.0008017, 0.0003392, 0.0007993, -0.0002445, 0.0002662
4: 0.0107541, 0.0132650, 0.0107680, 0.0133660, -0.0015031, 0.0013808
5: 0.9984940, 0.9991917, 0.9984980, 0.9992197, -0.0004176, 0.0003836
6: 0.0065167, 0.0071499, 0.0065202, 0.0071754, -0.0003791, 0.0003482
7: 0.0009377, 0.0033007, 0.0009508, 0.0033958, -0.0014146, 0.0012995
8: -0.0117618, -0.0099227, -0.0118358, -0.0099329, -0.0010114, 0.0011010
9: -0.0031537, -0.0029950, -0.0031528, -0.0029886, -0.0000950, 0.0000873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002339, upper bound: 0.0002303
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002339, upper bound: 0.0002303
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127277, -0.0111125, -0.0128187, -0.0111255, -0.0008835, 0.0009911
1: -0.0065271, -0.0060717, -0.0065527, -0.0060753, -0.0002491, 0.0002794
2: -0.0095983, -0.0062385, -0.0097876, -0.0062654, -0.0018379, 0.0020618
3: 0.0003571, 0.0008017, 0.0003321, 0.0007982, -0.0002432, 0.0002728
4: 0.0107541, 0.0132650, 0.0107742, 0.0134065, -0.0015408, 0.0013735
5: 0.9984940, 0.9991917, 0.9984996, 0.9992309, -0.0004281, 0.0003816
6: 0.0065167, 0.0071499, 0.0065218, 0.0071856, -0.0003886, 0.0003464
7: 0.0009377, 0.0033007, 0.0009566, 0.0034339, -0.0014501, 0.0012927
8: -0.0117618, -0.0099227, -0.0118654, -0.0099374, -0.0010061, 0.0011286
9: -0.0031537, -0.0029950, -0.0031524, -0.0029860, -0.0000974, 0.0000868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002339, upper bound: 0.0002303
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002339, upper bound: 0.0002303
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0126930, -0.0110927, -0.0127937, -0.0111211, -0.0008366, 0.0009774
1: -0.0065173, -0.0060661, -0.0065457, -0.0060741, -0.0002359, 0.0002756
2: -0.0095261, -0.0061972, -0.0097357, -0.0062563, -0.0017404, 0.0020333
3: 0.0003667, 0.0008072, 0.0003389, 0.0007994, -0.0002303, 0.0002691
4: 0.0107232, 0.0132111, 0.0107674, 0.0133677, -0.0015195, 0.0013006
5: 0.9984855, 0.9991767, 0.9984977, 0.9992202, -0.0004222, 0.0003614
6: 0.0065089, 0.0071363, 0.0065201, 0.0071758, -0.0003832, 0.0003280
7: 0.0009086, 0.0032499, 0.0009502, 0.0033974, -0.0014300, 0.0012240
8: -0.0117223, -0.0099001, -0.0118370, -0.0099324, -0.0009527, 0.0011130
9: -0.0031556, -0.0029984, -0.0031528, -0.0029885, -0.0000960, 0.0000822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002277, upper bound: 0.0002396
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002277, upper bound: 0.0002396
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0126930, -0.0110927, -0.0128246, -0.0111256, -0.0008048, 0.0009830
1: -0.0065173, -0.0060661, -0.0065544, -0.0060754, -0.0002269, 0.0002771
2: -0.0095261, -0.0061972, -0.0097999, -0.0062656, -0.0016742, 0.0020448
3: 0.0003667, 0.0008072, 0.0003304, 0.0007981, -0.0002216, 0.0002706
4: 0.0107232, 0.0132111, 0.0107744, 0.0134157, -0.0015281, 0.0012512
5: 0.9984855, 0.9991767, 0.9984996, 0.9992335, -0.0004246, 0.0003476
6: 0.0065089, 0.0071363, 0.0065218, 0.0071879, -0.0003854, 0.0003155
7: 0.0009086, 0.0032499, 0.0009567, 0.0034425, -0.0014381, 0.0011775
8: -0.0117223, -0.0099001, -0.0118722, -0.0099375, -0.0009165, 0.0011193
9: -0.0031556, -0.0029984, -0.0031524, -0.0029855, -0.0000966, 0.0000791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002277, upper bound: 0.0002445
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002277, upper bound: 0.0002445
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127554, -0.0111218, -0.0127937, -0.0111211, -0.0009248, 0.0009663
1: -0.0065349, -0.0060743, -0.0065457, -0.0060741, -0.0002607, 0.0002724
2: -0.0096560, -0.0062577, -0.0097357, -0.0062563, -0.0019237, 0.0020101
3: 0.0003495, 0.0007992, 0.0003389, 0.0007994, -0.0002546, 0.0002660
4: 0.0107685, 0.0133081, 0.0107674, 0.0133677, -0.0015022, 0.0014377
5: 0.9984980, 0.9992037, 0.9984977, 0.9992202, -0.0004174, 0.0003994
6: 0.0065203, 0.0071608, 0.0065201, 0.0071758, -0.0003788, 0.0003626
7: 0.0009512, 0.0033413, 0.0009502, 0.0033974, -0.0014137, 0.0013530
8: -0.0117934, -0.0099332, -0.0118370, -0.0099324, -0.0010530, 0.0011003
9: -0.0031527, -0.0029923, -0.0031528, -0.0029885, -0.0000949, 0.0000909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002277, upper bound: 0.0002473
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002277, upper bound: 0.0002473
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127554, -0.0111218, -0.0128246, -0.0111256, -0.0008970, 0.0009744
1: -0.0065349, -0.0060743, -0.0065544, -0.0060754, -0.0002529, 0.0002747
2: -0.0096560, -0.0062577, -0.0097999, -0.0062656, -0.0018660, 0.0020270
3: 0.0003495, 0.0007992, 0.0003304, 0.0007981, -0.0002469, 0.0002682
4: 0.0107685, 0.0133081, 0.0107744, 0.0134157, -0.0015148, 0.0013945
5: 0.9984980, 0.9992037, 0.9984996, 0.9992335, -0.0004209, 0.0003874
6: 0.0065203, 0.0071608, 0.0065218, 0.0071879, -0.0003820, 0.0003517
7: 0.0009512, 0.0033413, 0.0009567, 0.0034425, -0.0014256, 0.0013124
8: -0.0117934, -0.0099332, -0.0118722, -0.0099375, -0.0010214, 0.0011096
9: -0.0031527, -0.0029923, -0.0031524, -0.0029855, -0.0000957, 0.0000881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002277, upper bound: 0.0002477
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002277, upper bound: 0.0002477
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0126912, -0.0110904, -0.0127926, -0.0111215, -0.0008319, 0.0009736
1: -0.0065168, -0.0060654, -0.0065454, -0.0060742, -0.0002345, 0.0002745
2: -0.0095225, -0.0061924, -0.0097334, -0.0062571, -0.0017305, 0.0020252
3: 0.0003671, 0.0008078, 0.0003392, 0.0007993, -0.0002290, 0.0002680
4: 0.0107197, 0.0132084, 0.0107680, 0.0133660, -0.0015135, 0.0012933
5: 0.9984844, 0.9991760, 0.9984980, 0.9992197, -0.0004205, 0.0003593
6: 0.0065080, 0.0071356, 0.0065202, 0.0071754, -0.0003817, 0.0003261
7: 0.0009053, 0.0032474, 0.0009508, 0.0033958, -0.0014244, 0.0012171
8: -0.0117204, -0.0098974, -0.0118358, -0.0099329, -0.0009473, 0.0011086
9: -0.0031558, -0.0029986, -0.0031528, -0.0029886, -0.0000956, 0.0000817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002330, upper bound: 0.0002255
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002330, upper bound: 0.0002255
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0127550, -0.0111155, -0.0127926, -0.0111215, -0.0009241, 0.0009651
1: -0.0065348, -0.0060725, -0.0065454, -0.0060742, -0.0002605, 0.0002721
2: -0.0096551, -0.0062447, -0.0097334, -0.0062571, -0.0019223, 0.0020076
3: 0.0003496, 0.0008009, 0.0003392, 0.0007993, -0.0002544, 0.0002657
4: 0.0107588, 0.0133075, 0.0107680, 0.0133660, -0.0015004, 0.0014366
5: 0.9984954, 0.9992034, 0.9984980, 0.9992197, -0.0004169, 0.0003991
6: 0.0065179, 0.0071606, 0.0065202, 0.0071754, -0.0003784, 0.0003623
7: 0.0009421, 0.0033407, 0.0009508, 0.0033958, -0.0014120, 0.0013520
8: -0.0117929, -0.0099261, -0.0118358, -0.0099329, -0.0010523, 0.0010990
9: -0.0031534, -0.0029923, -0.0031528, -0.0029886, -0.0000948, 0.0000908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002330, upper bound: 0.0002255
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002330, upper bound: 0.0002255
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0127188, -0.0110975, -0.0128037, -0.0111212, -0.0008441, 0.0009828
1: -0.0065246, -0.0060675, -0.0065485, -0.0060741, -0.0002380, 0.0002771
2: -0.0095798, -0.0062072, -0.0097564, -0.0062565, -0.0017560, 0.0020444
3: 0.0003596, 0.0008059, 0.0003362, 0.0007993, -0.0002324, 0.0002705
4: 0.0107307, 0.0132512, 0.0107676, 0.0133832, -0.0015279, 0.0013123
5: 0.9984875, 0.9991878, 0.9984978, 0.9992245, -0.0004245, 0.0003646
6: 0.0065108, 0.0071464, 0.0065201, 0.0071797, -0.0003853, 0.0003309
7: 0.0009157, 0.0032877, 0.0009504, 0.0034119, -0.0014379, 0.0012350
8: -0.0117517, -0.0099055, -0.0118484, -0.0099326, -0.0009612, 0.0011191
9: -0.0031551, -0.0029959, -0.0031528, -0.0029875, -0.0000966, 0.0000829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002273, upper bound: 0.0002474
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002273, upper bound: 0.0002514
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0127830, -0.0111239, -0.0128037, -0.0111212, -0.0009363, 0.0009740
1: -0.0065427, -0.0060749, -0.0065485, -0.0060741, -0.0002640, 0.0002746
2: -0.0097134, -0.0062621, -0.0097564, -0.0062565, -0.0019477, 0.0020262
3: 0.0003419, 0.0007986, 0.0003362, 0.0007993, -0.0002577, 0.0002681
4: 0.0107717, 0.0133511, 0.0107676, 0.0133832, -0.0015143, 0.0014556
5: 0.9984990, 0.9992156, 0.9984978, 0.9992245, -0.0004207, 0.0004044
6: 0.0065211, 0.0071716, 0.0065201, 0.0071797, -0.0003819, 0.0003671
7: 0.0009543, 0.0033817, 0.0009504, 0.0034119, -0.0014251, 0.0013699
8: -0.0118249, -0.0099356, -0.0118484, -0.0099326, -0.0010662, 0.0011091
9: -0.0031525, -0.0029895, -0.0031528, -0.0029875, -0.0000957, 0.0000920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002273, upper bound: 0.0002549
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002273, upper bound: 0.0002551
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0126912, -0.0110904, -0.0128187, -0.0111255, -0.0008040, 0.0009854
1: -0.0065168, -0.0060654, -0.0065527, -0.0060753, -0.0002267, 0.0002778
2: -0.0095225, -0.0061924, -0.0097876, -0.0062654, -0.0016724, 0.0020498
3: 0.0003671, 0.0008078, 0.0003321, 0.0007982, -0.0002213, 0.0002713
4: 0.0107197, 0.0132084, 0.0107742, 0.0134065, -0.0015319, 0.0012498
5: 0.9984844, 0.9991760, 0.9984996, 0.9992309, -0.0004256, 0.0003472
6: 0.0065080, 0.0071356, 0.0065218, 0.0071856, -0.0003863, 0.0003152
7: 0.0009053, 0.0032474, 0.0009566, 0.0034339, -0.0014416, 0.0011762
8: -0.0117204, -0.0098974, -0.0118654, -0.0099374, -0.0009155, 0.0011220
9: -0.0031558, -0.0029986, -0.0031524, -0.0029860, -0.0000968, 0.0000790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002368, upper bound: 0.0002275
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002368, upper bound: 0.0002275
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0127550, -0.0111155, -0.0128187, -0.0111255, -0.0008956, 0.0009748
1: -0.0065348, -0.0060725, -0.0065527, -0.0060753, -0.0002525, 0.0002748
2: -0.0096551, -0.0062447, -0.0097876, -0.0062654, -0.0018630, 0.0020279
3: 0.0003496, 0.0008009, 0.0003321, 0.0007982, -0.0002465, 0.0002684
4: 0.0107588, 0.0133075, 0.0107742, 0.0134065, -0.0015155, 0.0013923
5: 0.9984954, 0.9992034, 0.9984996, 0.9992309, -0.0004210, 0.0003868
6: 0.0065179, 0.0071606, 0.0065218, 0.0071856, -0.0003822, 0.0003511
7: 0.0009421, 0.0033407, 0.0009566, 0.0034339, -0.0014262, 0.0013103
8: -0.0117929, -0.0099261, -0.0118654, -0.0099374, -0.0010198, 0.0011100
9: -0.0031534, -0.0029923, -0.0031524, -0.0029860, -0.0000958, 0.0000880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002368, upper bound: 0.0002418
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002368, upper bound: 0.0002418
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0127188, -0.0110975, -0.0128298, -0.0111252, -0.0008165, 0.0009966
1: -0.0065246, -0.0060675, -0.0065559, -0.0060753, -0.0002302, 0.0002810
2: -0.0095798, -0.0062072, -0.0098108, -0.0062648, -0.0016985, 0.0020731
3: 0.0003596, 0.0008059, 0.0003290, 0.0007983, -0.0002248, 0.0002743
4: 0.0107307, 0.0132512, 0.0107738, 0.0134239, -0.0015493, 0.0012693
5: 0.9984875, 0.9991878, 0.9984995, 0.9992357, -0.0004305, 0.0003527
6: 0.0065108, 0.0071464, 0.0065217, 0.0071900, -0.0003907, 0.0003201
7: 0.0009157, 0.0032877, 0.0009562, 0.0034502, -0.0014581, 0.0011946
8: -0.0117517, -0.0099055, -0.0118782, -0.0099371, -0.0009298, 0.0011348
9: -0.0031551, -0.0029959, -0.0031524, -0.0029849, -0.0000979, 0.0000802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002484
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002515
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0127830, -0.0111239, -0.0128298, -0.0111252, -0.0009081, 0.0009840
1: -0.0065427, -0.0060749, -0.0065559, -0.0060753, -0.0002560, 0.0002774
2: -0.0097134, -0.0062621, -0.0098108, -0.0062648, -0.0018891, 0.0020469
3: 0.0003419, 0.0007986, 0.0003290, 0.0007983, -0.0002500, 0.0002709
4: 0.0107717, 0.0133511, 0.0107738, 0.0134239, -0.0015297, 0.0014118
5: 0.9984990, 0.9992156, 0.9984995, 0.9992357, -0.0004250, 0.0003922
6: 0.0065211, 0.0071716, 0.0065217, 0.0071900, -0.0003858, 0.0003560
7: 0.0009543, 0.0033817, 0.0009562, 0.0034502, -0.0014397, 0.0013286
8: -0.0118249, -0.0099356, -0.0118782, -0.0099371, -0.0010341, 0.0011205
9: -0.0031525, -0.0029895, -0.0031524, -0.0029849, -0.0000967, 0.0000892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002549
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002552
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0126913, -0.0110924, -0.0128233, -0.0111456, -0.0008075, 0.0010123
1: -0.0065168, -0.0060660, -0.0065540, -0.0060810, -0.0002277, 0.0002854
2: -0.0095226, -0.0061967, -0.0097972, -0.0063073, -0.0016798, 0.0021057
3: 0.0003671, 0.0008073, 0.0003308, 0.0007926, -0.0002223, 0.0002787
4: 0.0107229, 0.0132085, 0.0108055, 0.0134137, -0.0015737, 0.0012554
5: 0.9984854, 0.9991759, 0.9985083, 0.9992330, -0.0004372, 0.0003488
6: 0.0065088, 0.0071357, 0.0065297, 0.0071874, -0.0003969, 0.0003166
7: 0.0009083, 0.0032475, 0.0009861, 0.0034406, -0.0014810, 0.0011814
8: -0.0117204, -0.0098998, -0.0118707, -0.0099603, -0.0009195, 0.0011527
9: -0.0031556, -0.0029986, -0.0031504, -0.0029856, -0.0000994, 0.0000793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002386, upper bound: 0.0002236
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002386, upper bound: 0.0002396
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0126913, -0.0110924, -0.0128492, -0.0111510, -0.0008023, 0.0010372
1: -0.0065168, -0.0060660, -0.0065613, -0.0060825, -0.0002262, 0.0002924
2: -0.0095226, -0.0061967, -0.0098511, -0.0063185, -0.0016689, 0.0021576
3: 0.0003671, 0.0008073, 0.0003237, 0.0007911, -0.0002209, 0.0002855
4: 0.0107229, 0.0132085, 0.0108139, 0.0134539, -0.0016124, 0.0012472
5: 0.9984854, 0.9991759, 0.9985107, 0.9992442, -0.0004480, 0.0003465
6: 0.0065088, 0.0071357, 0.0065318, 0.0071976, -0.0004066, 0.0003145
7: 0.0009083, 0.0032475, 0.0009940, 0.0034785, -0.0015175, 0.0011738
8: -0.0117204, -0.0098998, -0.0119002, -0.0099665, -0.0009136, 0.0011811
9: -0.0031556, -0.0029986, -0.0031499, -0.0029830, -0.0001019, 0.0000788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002386, upper bound: 0.0002236
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002386, upper bound: 0.0002396
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0126650, -0.0110851, -0.0128820, -0.0111556, -0.0007735, 0.0010688
1: -0.0065094, -0.0060640, -0.0065706, -0.0060838, -0.0002181, 0.0003013
2: -0.0094679, -0.0061813, -0.0099193, -0.0063280, -0.0016091, 0.0022233
3: 0.0003744, 0.0008093, 0.0003146, 0.0007899, -0.0002129, 0.0002942
4: 0.0107114, 0.0131676, 0.0108210, 0.0135049, -0.0016615, 0.0012026
5: 0.9984822, 0.9991646, 0.9985126, 0.9992583, -0.0004616, 0.0003341
6: 0.0065059, 0.0071253, 0.0065336, 0.0072104, -0.0004190, 0.0003033
7: 0.0008975, 0.0032090, 0.0010006, 0.0035265, -0.0015637, 0.0011317
8: -0.0116905, -0.0098914, -0.0119376, -0.0099717, -0.0008808, 0.0012170
9: -0.0031564, -0.0030011, -0.0031494, -0.0029798, -0.0001050, 0.0000760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B2_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002386, upper bound: 0.0002201
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002386, upper bound: 0.0002201
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0126930, -0.0110927, -0.0128820, -0.0111556, -0.0007824, 0.0010450
1: -0.0065173, -0.0060661, -0.0065706, -0.0060838, -0.0002206, 0.0002946
2: -0.0095261, -0.0061972, -0.0099193, -0.0063280, -0.0016275, 0.0021738
3: 0.0003667, 0.0008072, 0.0003146, 0.0007899, -0.0002154, 0.0002877
4: 0.0107232, 0.0132111, 0.0108210, 0.0135049, -0.0016245, 0.0012163
5: 0.9984855, 0.9991767, 0.9985126, 0.9992583, -0.0004513, 0.0003379
6: 0.0065089, 0.0071363, 0.0065336, 0.0072104, -0.0004097, 0.0003067
7: 0.0009086, 0.0032499, 0.0010006, 0.0035265, -0.0015289, 0.0011447
8: -0.0117223, -0.0099001, -0.0119376, -0.0099717, -0.0008909, 0.0011899
9: -0.0031556, -0.0029984, -0.0031494, -0.0029798, -0.0001027, 0.0000769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B2_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002386, upper bound: 0.0002441
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002386, upper bound: 0.0002442
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0127277, -0.0111125, -0.0128505, -0.0111496, -0.0008139, 0.0009933
1: -0.0065271, -0.0060717, -0.0065617, -0.0060821, -0.0002295, 0.0002801
2: -0.0095983, -0.0062385, -0.0098539, -0.0063155, -0.0016931, 0.0020663
3: 0.0003571, 0.0008017, 0.0003233, 0.0007915, -0.0002241, 0.0002734
4: 0.0107541, 0.0132650, 0.0108117, 0.0134560, -0.0015442, 0.0012653
5: 0.9984940, 0.9991917, 0.9985101, 0.9992447, -0.0004290, 0.0003515
6: 0.0065167, 0.0071499, 0.0065312, 0.0071981, -0.0003894, 0.0003191
7: 0.0009377, 0.0033007, 0.0009919, 0.0034805, -0.0014533, 0.0011908
8: -0.0117618, -0.0099227, -0.0119017, -0.0099648, -0.0009268, 0.0011311
9: -0.0031537, -0.0029950, -0.0031500, -0.0029829, -0.0000976, 0.0000800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A1_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002522, upper bound: 0.0002349
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_A1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002522, upper bound: 0.0002349
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0127277, -0.0111125, -0.0128764, -0.0111554, -0.0008095, 0.0010177
1: -0.0065271, -0.0060717, -0.0065690, -0.0060838, -0.0002282, 0.0002869
2: -0.0095983, -0.0062385, -0.0099077, -0.0063276, -0.0016840, 0.0021171
3: 0.0003571, 0.0008017, 0.0003162, 0.0007899, -0.0002229, 0.0002802
4: 0.0107541, 0.0132650, 0.0108207, 0.0134962, -0.0015822, 0.0012585
5: 0.9984940, 0.9991917, 0.9985126, 0.9992559, -0.0004396, 0.0003497
6: 0.0065167, 0.0071499, 0.0065335, 0.0072082, -0.0003990, 0.0003174
7: 0.0009377, 0.0033007, 0.0010004, 0.0035183, -0.0014890, 0.0011844
8: -0.0117618, -0.0099227, -0.0119312, -0.0099715, -0.0009218, 0.0011589
9: -0.0031537, -0.0029950, -0.0031494, -0.0029804, -0.0001000, 0.0000795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002522, upper bound: 0.0002348
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002522, upper bound: 0.0002349
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127554, -0.0111218, -0.0128517, -0.0111509, -0.0008546, 0.0009947
1: -0.0065349, -0.0060743, -0.0065620, -0.0060825, -0.0002409, 0.0002805
2: -0.0096560, -0.0062577, -0.0098563, -0.0063184, -0.0017777, 0.0020693
3: 0.0003495, 0.0007992, 0.0003230, 0.0007912, -0.0002353, 0.0002738
4: 0.0107685, 0.0133081, 0.0108138, 0.0134578, -0.0015464, 0.0013286
5: 0.9984980, 0.9992037, 0.9985106, 0.9992452, -0.0004296, 0.0003691
6: 0.0065203, 0.0071608, 0.0065318, 0.0071985, -0.0003900, 0.0003350
7: 0.0009512, 0.0033413, 0.0009939, 0.0034822, -0.0014554, 0.0012503
8: -0.0117934, -0.0099332, -0.0119031, -0.0099664, -0.0009731, 0.0011327
9: -0.0031527, -0.0029923, -0.0031499, -0.0029828, -0.0000977, 0.0000840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B2_A1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002367, upper bound: 0.0002478
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002367, upper bound: 0.0002478
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127554, -0.0111218, -0.0128820, -0.0111556, -0.0008238, 0.0010015
1: -0.0065349, -0.0060743, -0.0065706, -0.0060838, -0.0002323, 0.0002824
2: -0.0096560, -0.0062577, -0.0099193, -0.0063280, -0.0017137, 0.0020833
3: 0.0003495, 0.0007992, 0.0003146, 0.0007899, -0.0002268, 0.0002757
4: 0.0107685, 0.0133081, 0.0108210, 0.0135049, -0.0015569, 0.0012807
5: 0.9984980, 0.9992037, 0.9985126, 0.9992583, -0.0004326, 0.0003558
6: 0.0065203, 0.0071608, 0.0065336, 0.0072104, -0.0003926, 0.0003230
7: 0.0009512, 0.0033413, 0.0010006, 0.0035265, -0.0014652, 0.0012053
8: -0.0117934, -0.0099332, -0.0119376, -0.0099717, -0.0009381, 0.0011404
9: -0.0031527, -0.0029923, -0.0031494, -0.0029798, -0.0000984, 0.0000809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B2_A1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002367, upper bound: 0.0002479
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002367, upper bound: 0.0002480
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0127168, -0.0110973, -0.0128233, -0.0111456, -0.0008417, 0.0010078
1: -0.0065240, -0.0060674, -0.0065540, -0.0060810, -0.0002373, 0.0002841
2: -0.0095757, -0.0062067, -0.0097972, -0.0063073, -0.0017509, 0.0020964
3: 0.0003601, 0.0008059, 0.0003308, 0.0007926, -0.0002317, 0.0002774
4: 0.0107304, 0.0132482, 0.0108055, 0.0134137, -0.0015667, 0.0013085
5: 0.9984875, 0.9991870, 0.9985083, 0.9992330, -0.0004353, 0.0003635
6: 0.0065107, 0.0071457, 0.0065297, 0.0071874, -0.0003951, 0.0003300
7: 0.0009153, 0.0032849, 0.0009861, 0.0034406, -0.0014745, 0.0012315
8: -0.0117495, -0.0099053, -0.0118707, -0.0099603, -0.0009584, 0.0011476
9: -0.0031552, -0.0029960, -0.0031504, -0.0029856, -0.0000990, 0.0000827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002353, upper bound: 0.0002255
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002353, upper bound: 0.0002472
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0127273, -0.0110970, -0.0128534, -0.0111498, -0.0008490, 0.0010155
1: -0.0065270, -0.0060673, -0.0065625, -0.0060822, -0.0002394, 0.0002863
2: -0.0095975, -0.0062062, -0.0098598, -0.0063160, -0.0017661, 0.0021124
3: 0.0003572, 0.0008060, 0.0003225, 0.0007915, -0.0002337, 0.0002795
4: 0.0107300, 0.0132644, 0.0108121, 0.0134605, -0.0015787, 0.0013199
5: 0.9984874, 0.9991915, 0.9985102, 0.9992460, -0.0004386, 0.0003667
6: 0.0065106, 0.0071498, 0.0065313, 0.0071992, -0.0003981, 0.0003329
7: 0.0009150, 0.0033002, 0.0009922, 0.0034847, -0.0014857, 0.0012421
8: -0.0117614, -0.0099050, -0.0119050, -0.0099651, -0.0009668, 0.0011563
9: -0.0031552, -0.0029950, -0.0031500, -0.0029826, -0.0000998, 0.0000834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002571, upper bound: 0.0002255
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002571, upper bound: 0.0002497
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0127168, -0.0110973, -0.0128492, -0.0111510, -0.0008152, 0.0010219
1: -0.0065240, -0.0060674, -0.0065613, -0.0060825, -0.0002298, 0.0002881
2: -0.0095757, -0.0062067, -0.0098511, -0.0063185, -0.0016957, 0.0021257
3: 0.0003601, 0.0008059, 0.0003237, 0.0007911, -0.0002244, 0.0002813
4: 0.0107304, 0.0132482, 0.0108139, 0.0134539, -0.0015886, 0.0012673
5: 0.9984875, 0.9991870, 0.9985107, 0.9992442, -0.0004414, 0.0003521
6: 0.0065107, 0.0071457, 0.0065318, 0.0071976, -0.0004006, 0.0003196
7: 0.0009153, 0.0032849, 0.0009940, 0.0034785, -0.0014951, 0.0011927
8: -0.0117495, -0.0099053, -0.0119002, -0.0099665, -0.0009282, 0.0011636
9: -0.0031552, -0.0029960, -0.0031499, -0.0029830, -0.0001004, 0.0000801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002371, upper bound: 0.0002275
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002371, upper bound: 0.0002480
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0127273, -0.0110970, -0.0128791, -0.0111556, -0.0008228, 0.0010275
1: -0.0065270, -0.0060673, -0.0065698, -0.0060838, -0.0002320, 0.0002897
2: -0.0095975, -0.0062062, -0.0099133, -0.0063281, -0.0017115, 0.0021373
3: 0.0003572, 0.0008060, 0.0003154, 0.0007899, -0.0002265, 0.0002828
4: 0.0107300, 0.0132644, 0.0108211, 0.0135005, -0.0015973, 0.0012791
5: 0.9984874, 0.9991915, 0.9985126, 0.9992571, -0.0004438, 0.0003554
6: 0.0065106, 0.0071498, 0.0065336, 0.0072093, -0.0004028, 0.0003226
7: 0.0009150, 0.0033002, 0.0010007, 0.0035223, -0.0015033, 0.0012038
8: -0.0117614, -0.0099050, -0.0119343, -0.0099717, -0.0009369, 0.0011700
9: -0.0031552, -0.0029950, -0.0031494, -0.0029801, -0.0001009, 0.0000808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002572, upper bound: 0.0002275
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002572, upper bound: 0.0002498
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0127550, -0.0111155, -0.0128505, -0.0111496, -0.0008496, 0.0009906
1: -0.0065348, -0.0060725, -0.0065617, -0.0060821, -0.0002395, 0.0002793
2: -0.0096551, -0.0062447, -0.0098539, -0.0063155, -0.0017673, 0.0020606
3: 0.0003496, 0.0008009, 0.0003233, 0.0007915, -0.0002339, 0.0002727
4: 0.0107588, 0.0133075, 0.0108117, 0.0134560, -0.0015400, 0.0013207
5: 0.9984954, 0.9992034, 0.9985101, 0.9992447, -0.0004278, 0.0003669
6: 0.0065179, 0.0071606, 0.0065312, 0.0071981, -0.0003884, 0.0003331
7: 0.0009421, 0.0033407, 0.0009919, 0.0034805, -0.0014493, 0.0012430
8: -0.0117929, -0.0099261, -0.0119017, -0.0099648, -0.0009674, 0.0011280
9: -0.0031534, -0.0029923, -0.0031500, -0.0029829, -0.0000973, 0.0000835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002345, upper bound: 0.0002425
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002345, upper bound: 0.0002426
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127830, -0.0111239, -0.0128615, -0.0111492, -0.0008630, 0.0009997
1: -0.0065427, -0.0060749, -0.0065648, -0.0060820, -0.0002433, 0.0002819
2: -0.0097134, -0.0062621, -0.0098766, -0.0063148, -0.0017951, 0.0020796
3: 0.0003419, 0.0007986, 0.0003203, 0.0007916, -0.0002376, 0.0002752
4: 0.0107717, 0.0133511, 0.0108112, 0.0134731, -0.0015541, 0.0013416
5: 0.9984990, 0.9992156, 0.9985099, 0.9992495, -0.0004318, 0.0003727
6: 0.0065211, 0.0071716, 0.0065311, 0.0072024, -0.0003919, 0.0003383
7: 0.0009543, 0.0033817, 0.0009914, 0.0034965, -0.0014626, 0.0012626
8: -0.0118249, -0.0099356, -0.0119142, -0.0099645, -0.0009827, 0.0011384
9: -0.0031525, -0.0029895, -0.0031501, -0.0029818, -0.0000982, 0.0000848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002345, upper bound: 0.0002552
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002345, upper bound: 0.0002554
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0127550, -0.0111155, -0.0128764, -0.0111554, -0.0008225, 0.0010031
1: -0.0065348, -0.0060725, -0.0065690, -0.0060838, -0.0002319, 0.0002828
2: -0.0096551, -0.0062447, -0.0099077, -0.0063276, -0.0017109, 0.0020866
3: 0.0003496, 0.0008009, 0.0003162, 0.0007899, -0.0002264, 0.0002761
4: 0.0107588, 0.0133075, 0.0108207, 0.0134962, -0.0015594, 0.0012786
5: 0.9984954, 0.9992034, 0.9985126, 0.9992559, -0.0004332, 0.0003552
6: 0.0065179, 0.0071606, 0.0065335, 0.0072082, -0.0003933, 0.0003225
7: 0.0009421, 0.0033407, 0.0010004, 0.0035183, -0.0014676, 0.0012033
8: -0.0117929, -0.0099261, -0.0119312, -0.0099715, -0.0009366, 0.0011422
9: -0.0031534, -0.0029923, -0.0031494, -0.0029804, -0.0000985, 0.0000808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002348, upper bound: 0.0002428
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002348, upper bound: 0.0002426
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127830, -0.0111239, -0.0128873, -0.0111551, -0.0008356, 0.0010138
1: -0.0065427, -0.0060749, -0.0065721, -0.0060837, -0.0002356, 0.0002858
2: -0.0097134, -0.0062621, -0.0099304, -0.0063269, -0.0017381, 0.0021090
3: 0.0003419, 0.0007986, 0.0003132, 0.0007900, -0.0002300, 0.0002791
4: 0.0107717, 0.0133511, 0.0108202, 0.0135132, -0.0015761, 0.0012990
5: 0.9984990, 0.9992156, 0.9985124, 0.9992606, -0.0004379, 0.0003609
6: 0.0065211, 0.0071716, 0.0065334, 0.0072125, -0.0003975, 0.0003276
7: 0.0009543, 0.0033817, 0.0009999, 0.0035343, -0.0014833, 0.0012225
8: -0.0118249, -0.0099356, -0.0119436, -0.0099711, -0.0009515, 0.0011545
9: -0.0031525, -0.0029895, -0.0031495, -0.0029793, -0.0000996, 0.0000821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.98 + 598.75 = 601.73 seconds
