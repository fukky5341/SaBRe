## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.06656274


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0167351, 0.0122514, -0.0167351, 0.0122514, -0.0289865, 0.0289865)
1: (-0.0293941, 0.0100843, -0.0293941, 0.0100843, -0.0394784, 0.0394784)
2: (0.0287971, 0.0648342, 0.0287971, 0.0648342, -0.0360371, 0.0360371)
3: (-0.0009592, 0.0532627, -0.0009592, 0.0532627, -0.0473836, 0.0473836)
4: (-0.0228520, 0.0217762, -0.0228520, 0.0217762, -0.0446282, 0.0446282)
5: (-0.0059717, 0.0378844, -0.0059717, 0.0378844, -0.0438561, 0.0438561)
6: (-0.0481563, -0.0068403, -0.0481563, -0.0068403, -0.0413160, 0.0413160)
7: (0.8629556, 0.9693236, 0.8629556, 0.9693236, -0.1063680, 0.1063680)
8: (-0.0103393, 0.0420309, -0.0103393, 0.0420309, -0.0523702, 0.0523702)
9: (-0.0205963, 0.0211083, -0.0205963, 0.0211083, -0.0417046, 0.0417046)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.53 + 1.48 = 3.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0693844, upper bound: 0.0693844

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683399, upper bound: 0.0680776
time: 0.61 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680791, upper bound: 0.0680791
time: 0.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.38 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 7, lower bound: -0.0683399, upper bound: 0.0680776
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 7, lower bound: -0.0680791, upper bound: 0.0680791

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0167333, 0.0114246, -0.0167351, 0.0122514, -0.0289847, 0.0281596
1: -0.0274564, 0.0089185, -0.0293941, 0.0100843, -0.0375407, 0.0383126
2: 0.0299972, 0.0634103, 0.0287971, 0.0648342, -0.0348369, 0.0346132
3: -0.0009561, 0.0514151, -0.0009592, 0.0532627, -0.0472234, 0.0455218
4: -0.0211238, 0.0194287, -0.0228520, 0.0217762, -0.0429000, 0.0422807
5: -0.0040470, 0.0357544, -0.0059717, 0.0378844, -0.0419314, 0.0417262
6: -0.0463009, -0.0068430, -0.0481563, -0.0068403, -0.0394606, 0.0413133
7: 0.8661445, 0.9693184, 0.8629556, 0.9693236, -0.1031791, 0.1063628
8: -0.0103361, 0.0398122, -0.0103393, 0.0420309, -0.0523670, 0.0501516
9: -0.0197753, 0.0175714, -0.0205963, 0.0211083, -0.0408835, 0.0381677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680776, upper bound: 0.0680776
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680776, upper bound: 0.0680776
time: 0.67 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0169213, 0.0109802, -0.0167346, 0.0120423, -0.0289636, 0.0277148
1: -0.0264151, 0.0082920, -0.0289042, 0.0097895, -0.0362047, 0.0371962
2: 0.0306422, 0.0626451, 0.0291006, 0.0644742, -0.0338320, 0.0335446
3: -0.0012954, 0.0504225, -0.0009584, 0.0527956, -0.0471439, 0.0448211
4: -0.0201950, 0.0181672, -0.0224150, 0.0211827, -0.0413777, 0.0405822
5: -0.0030127, 0.0346098, -0.0054851, 0.0373459, -0.0403586, 0.0400949
6: -0.0453039, -0.0065541, -0.0476872, -0.0068410, -0.0384629, 0.0411331
7: 0.8678581, 0.9698745, 0.8637618, 0.9693221, -0.1014640, 0.1061127
8: -0.0106800, 0.0386200, -0.0103384, 0.0414699, -0.0521499, 0.0489584
9: -0.0193341, 0.0156707, -0.0203887, 0.0202140, -0.0395480, 0.0360595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680776, upper bound: 0.0680791
time: 0.55 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680776, upper bound: 0.0680791
time: 0.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.67 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 7, lower bound: -0.0680776, upper bound: 0.0680776
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 7, lower bound: -0.0680776, upper bound: 0.0680776
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 7, lower bound: -0.0680776, upper bound: 0.0680791
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 7, lower bound: -0.0680776, upper bound: 0.0680791

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0167333, 0.0114246, -0.0167333, 0.0114246, -0.0281579, 0.0281579
1: -0.0274564, 0.0089185, -0.0274564, 0.0089185, -0.0363749, 0.0363749
2: 0.0299972, 0.0634103, 0.0299972, 0.0634103, -0.0334131, 0.0334131
3: -0.0009561, 0.0514151, -0.0009561, 0.0514151, -0.0453615, 0.0453615
4: -0.0211238, 0.0194287, -0.0211238, 0.0194287, -0.0405524, 0.0405524
5: -0.0040470, 0.0357544, -0.0040470, 0.0357544, -0.0398014, 0.0398014
6: -0.0463009, -0.0068430, -0.0463009, -0.0068430, -0.0394580, 0.0394580
7: 0.8661445, 0.9693184, 0.8661445, 0.9693184, -0.1031739, 0.1031739
8: -0.0103361, 0.0398122, -0.0103361, 0.0398122, -0.0501483, 0.0501483
9: -0.0197753, 0.0175714, -0.0197753, 0.0175714, -0.0373467, 0.0373467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673596, upper bound: 0.0675382
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0681451, upper bound: 0.0678697
time: 0.56 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0167333, 0.0114246, -0.0169213, 0.0109802, -0.0277135, 0.0283458
1: -0.0274564, 0.0089185, -0.0264151, 0.0082920, -0.0357484, 0.0353337
2: 0.0299972, 0.0634103, 0.0306422, 0.0626451, -0.0326479, 0.0327681
3: -0.0009561, 0.0514151, -0.0012954, 0.0504225, -0.0444797, 0.0457310
4: -0.0211238, 0.0194287, -0.0201950, 0.0181672, -0.0392909, 0.0396237
5: -0.0040470, 0.0357544, -0.0030127, 0.0346098, -0.0386568, 0.0387671
6: -0.0463009, -0.0068430, -0.0453039, -0.0065541, -0.0397469, 0.0384609
7: 0.8661445, 0.9693184, 0.8678581, 0.9698745, -0.1037300, 0.1014603
8: -0.0103361, 0.0398122, -0.0106800, 0.0386200, -0.0489561, 0.0504923
9: -0.0197753, 0.0175714, -0.0193341, 0.0156707, -0.0354460, 0.0369055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673596, upper bound: 0.0675382
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0681451, upper bound: 0.0678697
time: 0.57 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0169213, 0.0109802, -0.0167333, 0.0114246, -0.0283458, 0.0277135
1: -0.0264151, 0.0082920, -0.0274564, 0.0089185, -0.0353337, 0.0357484
2: 0.0306422, 0.0626451, 0.0299972, 0.0634103, -0.0327681, 0.0326479
3: -0.0012954, 0.0504225, -0.0009561, 0.0514151, -0.0457310, 0.0444797
4: -0.0201950, 0.0181672, -0.0211238, 0.0194287, -0.0396237, 0.0392909
5: -0.0030127, 0.0346098, -0.0040470, 0.0357544, -0.0387671, 0.0386568
6: -0.0453039, -0.0065541, -0.0463009, -0.0068430, -0.0384609, 0.0397469
7: 0.8678581, 0.9698745, 0.8661445, 0.9693184, -0.1014603, 0.1037300
8: -0.0106800, 0.0386200, -0.0103361, 0.0398122, -0.0504923, 0.0489561
9: -0.0193341, 0.0156707, -0.0197753, 0.0175714, -0.0369055, 0.0354460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0671474, upper bound: 0.0675425
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0678697, upper bound: 0.0678711
time: 0.73 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0169213, 0.0109802, -0.0169213, 0.0109802, -0.0279014, 0.0279014
1: -0.0264151, 0.0082920, -0.0264151, 0.0082920, -0.0347071, 0.0347071
2: 0.0306422, 0.0626451, 0.0306422, 0.0626451, -0.0320029, 0.0320029
3: -0.0012954, 0.0504225, -0.0012954, 0.0504225, -0.0447942, 0.0447942
4: -0.0201950, 0.0181672, -0.0201950, 0.0181672, -0.0383622, 0.0383622
5: -0.0030127, 0.0346098, -0.0030127, 0.0346098, -0.0376225, 0.0376225
6: -0.0453039, -0.0065541, -0.0453039, -0.0065541, -0.0387499, 0.0387499
7: 0.8678581, 0.9698745, 0.8678581, 0.9698745, -0.1020164, 0.1020164
8: -0.0106800, 0.0386200, -0.0106800, 0.0386200, -0.0493000, 0.0493000
9: -0.0193341, 0.0156707, -0.0193341, 0.0156707, -0.0350048, 0.0350048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0671474, upper bound: 0.0675425
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0678697, upper bound: 0.0678711
time: 0.57 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.73 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 7, lower bound: -0.0673596, upper bound: 0.0675382
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 7, lower bound: -0.0681451, upper bound: 0.0678697
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 7, lower bound: -0.0673596, upper bound: 0.0675382
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 7, lower bound: -0.0681451, upper bound: 0.0678697
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 7, lower bound: -0.0671474, upper bound: 0.0675425
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 7, lower bound: -0.0678697, upper bound: 0.0678711
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 7, lower bound: -0.0671474, upper bound: 0.0675425
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 7, lower bound: -0.0678697, upper bound: 0.0678711

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0167887, 0.0109225, -0.0167285, 0.0113193, -0.0281080, 0.0276510
1: -0.0262801, 0.0082108, -0.0272099, 0.0087701, -0.0350502, 0.0354207
2: 0.0307258, 0.0625459, 0.0301499, 0.0632291, -0.0325033, 0.0323959
3: -0.0010559, 0.0502936, -0.0009474, 0.0511802, -0.0450673, 0.0442496
4: -0.0200746, 0.0180035, -0.0209038, 0.0191300, -0.0392046, 0.0389073
5: -0.0028785, 0.0344613, -0.0038021, 0.0354834, -0.0383619, 0.0382635
6: -0.0451746, -0.0067579, -0.0460649, -0.0068504, -0.0383242, 0.0393070
7: 0.8680803, 0.9694820, 0.8665503, 0.9693042, -0.1012239, 0.1029317
8: -0.0104374, 0.0384653, -0.0103273, 0.0395300, -0.0499673, 0.0487926
9: -0.0192768, 0.0154242, -0.0196708, 0.0171213, -0.0363981, 0.0350950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0577679, upper bound: 0.0604432
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673449, upper bound: 0.0678074
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0167277, 0.0112405, -0.0167333, 0.0114246, -0.0281523, 0.0279738
1: -0.0270252, 0.0086591, -0.0274564, 0.0089185, -0.0359438, 0.0361155
2: 0.0302643, 0.0630935, 0.0299972, 0.0634103, -0.0331460, 0.0330962
3: -0.0009459, 0.0510042, -0.0009561, 0.0514151, -0.0452082, 0.0449529
4: -0.0207392, 0.0189063, -0.0211238, 0.0194287, -0.0401678, 0.0400301
5: -0.0036187, 0.0352805, -0.0040470, 0.0357544, -0.0393732, 0.0393275
6: -0.0458881, -0.0068516, -0.0463009, -0.0068430, -0.0390452, 0.0394493
7: 0.8668540, 0.9693018, 0.8661445, 0.9693184, -0.1024644, 0.1031573
8: -0.0103258, 0.0393186, -0.0103361, 0.0398122, -0.0501381, 0.0496547
9: -0.0195926, 0.0167844, -0.0197753, 0.0175714, -0.0371640, 0.0365596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679495, upper bound: 0.0674832
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679495, upper bound: 0.0682946
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0167887, 0.0109225, -0.0169164, 0.0108803, -0.0276690, 0.0278390
1: -0.0262801, 0.0082108, -0.0261812, 0.0081513, -0.0344314, 0.0343920
2: 0.0307258, 0.0625459, 0.0307871, 0.0624732, -0.0317474, 0.0317588
3: -0.0010559, 0.0502936, -0.0012867, 0.0501994, -0.0441861, 0.0446224
4: -0.0200746, 0.0180035, -0.0199864, 0.0178837, -0.0379583, 0.0379899
5: -0.0028785, 0.0344613, -0.0027803, 0.0343527, -0.0372312, 0.0372416
6: -0.0451746, -0.0067579, -0.0450799, -0.0065614, -0.0386131, 0.0383220
7: 0.8680803, 0.9694820, 0.8682433, 0.9698602, -0.1017799, 0.1012387
8: -0.0104374, 0.0384653, -0.0106712, 0.0383522, -0.0487895, 0.0491365
9: -0.0192768, 0.0154242, -0.0192349, 0.0152437, -0.0345205, 0.0346591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0541186, upper bound: 0.0550479
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0672252, upper bound: 0.0673999
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0167277, 0.0112405, -0.0169213, 0.0109802, -0.0277079, 0.0281618
1: -0.0270252, 0.0086591, -0.0264151, 0.0082920, -0.0353172, 0.0350742
2: 0.0302643, 0.0630935, 0.0306422, 0.0626451, -0.0323808, 0.0324513
3: -0.0009459, 0.0510042, -0.0012954, 0.0504225, -0.0443264, 0.0453191
4: -0.0207392, 0.0189063, -0.0201950, 0.0181672, -0.0389064, 0.0391013
5: -0.0036187, 0.0352805, -0.0030127, 0.0346098, -0.0382285, 0.0382932
6: -0.0458881, -0.0068516, -0.0453039, -0.0065541, -0.0393341, 0.0384523
7: 0.8668540, 0.9693018, 0.8678581, 0.9698745, -0.1030205, 0.1014437
8: -0.0103258, 0.0393186, -0.0106800, 0.0386200, -0.0489458, 0.0499986
9: -0.0195926, 0.0167844, -0.0193341, 0.0156707, -0.0352633, 0.0361184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0678355, upper bound: 0.0671474
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0678355, upper bound: 0.0678697
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0169960, 0.0104993, -0.0167285, 0.0113193, -0.0283153, 0.0272278
1: -0.0252882, 0.0076141, -0.0272099, 0.0087701, -0.0340584, 0.0348240
2: 0.0313402, 0.0618170, 0.0301499, 0.0632291, -0.0318890, 0.0316671
3: -0.0014304, 0.0493480, -0.0009474, 0.0511802, -0.0454732, 0.0433762
4: -0.0191899, 0.0168018, -0.0209038, 0.0191300, -0.0383199, 0.0377057
5: -0.0018933, 0.0333711, -0.0038021, 0.0354834, -0.0373766, 0.0371732
6: -0.0442249, -0.0064392, -0.0460649, -0.0068504, -0.0373745, 0.0396257
7: 0.8697128, 0.9700955, 0.8665503, 0.9693042, -0.0995914, 0.1035452
8: -0.0108167, 0.0373297, -0.0103273, 0.0395300, -0.0503467, 0.0476570
9: -0.0188566, 0.0136138, -0.0196708, 0.0171213, -0.0359779, 0.0332846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0553748, upper bound: 0.0592281
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0670149, upper bound: 0.0677004
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0169158, 0.0107942, -0.0167333, 0.0114246, -0.0283403, 0.0275275
1: -0.0259793, 0.0080298, -0.0274564, 0.0089185, -0.0348979, 0.0354862
2: 0.0309121, 0.0623249, 0.0299972, 0.0634103, -0.0324982, 0.0323277
3: -0.0012856, 0.0500069, -0.0009561, 0.0514151, -0.0455813, 0.0440485
4: -0.0198064, 0.0176392, -0.0211238, 0.0194287, -0.0392350, 0.0387629
5: -0.0025798, 0.0341308, -0.0040470, 0.0357544, -0.0383343, 0.0381778
6: -0.0448867, -0.0065625, -0.0463009, -0.0068430, -0.0380437, 0.0397385
7: 0.8685753, 0.9698582, 0.8661445, 0.9693184, -0.1007431, 0.1037137
8: -0.0106700, 0.0381210, -0.0103361, 0.0398122, -0.0504822, 0.0484571
9: -0.0191494, 0.0148753, -0.0197753, 0.0175714, -0.0367208, 0.0346506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0675382, upper bound: 0.0673596
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0675382, upper bound: 0.0673596
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0169960, 0.0104993, -0.0169164, 0.0108803, -0.0278763, 0.0274157
1: -0.0252882, 0.0076141, -0.0261812, 0.0081513, -0.0334395, 0.0337953
2: 0.0313402, 0.0618170, 0.0307871, 0.0624732, -0.0311331, 0.0310300
3: -0.0014304, 0.0493480, -0.0012867, 0.0501994, -0.0445349, 0.0436998
4: -0.0191899, 0.0168018, -0.0199864, 0.0178837, -0.0370737, 0.0367882
5: -0.0018933, 0.0333711, -0.0027803, 0.0343527, -0.0362459, 0.0361514
6: -0.0442249, -0.0064392, -0.0450799, -0.0065614, -0.0376635, 0.0386408
7: 0.8697128, 0.9700955, 0.8682433, 0.9698602, -0.1001474, 0.1018522
8: -0.0108167, 0.0373297, -0.0106712, 0.0383522, -0.0491689, 0.0480009
9: -0.0188566, 0.0136138, -0.0192349, 0.0152437, -0.0341002, 0.0328487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0533913, upper bound: 0.0550262
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0670149, upper bound: 0.0674064
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0169158, 0.0107942, -0.0169213, 0.0109802, -0.0278960, 0.0277155
1: -0.0259793, 0.0080298, -0.0264151, 0.0082920, -0.0342713, 0.0344450
2: 0.0309121, 0.0623249, 0.0306422, 0.0626451, -0.0317330, 0.0316827
3: -0.0012856, 0.0500069, -0.0012954, 0.0504225, -0.0446491, 0.0443767
4: -0.0198064, 0.0176392, -0.0201950, 0.0181672, -0.0379735, 0.0378342
5: -0.0025798, 0.0341308, -0.0030127, 0.0346098, -0.0371896, 0.0371435
6: -0.0448867, -0.0065625, -0.0453039, -0.0065541, -0.0383326, 0.0387415
7: 0.8685753, 0.9698582, 0.8678581, 0.9698745, -0.1012992, 0.1020001
8: -0.0106700, 0.0381210, -0.0106800, 0.0386200, -0.0492900, 0.0488011
9: -0.0191494, 0.0148753, -0.0193341, 0.0156707, -0.0348201, 0.0342094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0675382, upper bound: 0.0671480
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0675382, upper bound: 0.0678711
time: 0.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.88 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0577679, upper bound: 0.0604432
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0673449, upper bound: 0.0678074
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0679495, upper bound: 0.0674832
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0679495, upper bound: 0.0682946
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0541186, upper bound: 0.0550479
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0672252, upper bound: 0.0673999
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0678355, upper bound: 0.0671474
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0678355, upper bound: 0.0678697
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0553748, upper bound: 0.0592281
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0670149, upper bound: 0.0677004
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0675382, upper bound: 0.0673596
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0675382, upper bound: 0.0673596
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0533913, upper bound: 0.0550262
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0670149, upper bound: 0.0674064
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0675382, upper bound: 0.0671480
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 7, lower bound: -0.0675382, upper bound: 0.0678711

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0167887, 0.0109225, -0.0166705, 0.0113047, -0.0280933, 0.0275930
1: -0.0262801, 0.0082108, -0.0271755, 0.0087495, -0.0350296, 0.0353863
2: 0.0307258, 0.0625459, 0.0301712, 0.0632039, -0.0324781, 0.0323747
3: -0.0010559, 0.0502936, -0.0008426, 0.0511474, -0.0450369, 0.0440783
4: -0.0200746, 0.0180035, -0.0208732, 0.0190884, -0.0391630, 0.0388767
5: -0.0028785, 0.0344613, -0.0037680, 0.0354457, -0.0383242, 0.0382293
6: -0.0451746, -0.0067579, -0.0460320, -0.0069396, -0.0382350, 0.0392741
7: 0.8680803, 0.9694820, 0.8666067, 0.9691324, -0.1010520, 0.1028754
8: -0.0104374, 0.0384653, -0.0102211, 0.0394907, -0.0499280, 0.0486864
9: -0.0192768, 0.0154242, -0.0196563, 0.0170587, -0.0363355, 0.0350805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0540954, upper bound: 0.0545572
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0652719, upper bound: 0.0654074
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673213, upper bound: 0.0677832
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0167277, 0.0112405, -0.0167887, 0.0109225, -0.0276502, 0.0280292
1: -0.0270252, 0.0086591, -0.0262801, 0.0082108, -0.0352360, 0.0349392
2: 0.0302643, 0.0630935, 0.0307258, 0.0625459, -0.0322816, 0.0323676
3: -0.0009459, 0.0510042, -0.0010559, 0.0502936, -0.0441512, 0.0448594
4: -0.0207392, 0.0189063, -0.0200746, 0.0180035, -0.0387427, 0.0389809
5: -0.0036187, 0.0352805, -0.0028785, 0.0344613, -0.0380801, 0.0381590
6: -0.0458881, -0.0068516, -0.0451746, -0.0067579, -0.0391302, 0.0383230
7: 0.8668540, 0.9693018, 0.8680803, 0.9694820, -0.1026281, 0.1012215
8: -0.0103258, 0.0393186, -0.0104374, 0.0384653, -0.0487912, 0.0497559
9: -0.0195926, 0.0167844, -0.0192768, 0.0154242, -0.0350168, 0.0360612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0604432, upper bound: 0.0577679
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0678074, upper bound: 0.0673449
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0167277, 0.0112405, -0.0167277, 0.0112405, -0.0279682, 0.0279682
1: -0.0270252, 0.0086591, -0.0270252, 0.0086591, -0.0356843, 0.0356843
2: 0.0302643, 0.0630935, 0.0302643, 0.0630935, -0.0328292, 0.0328292
3: -0.0009459, 0.0510042, -0.0009459, 0.0510042, -0.0448206, 0.0448206
4: -0.0207392, 0.0189063, -0.0207392, 0.0189063, -0.0396455, 0.0396455
5: -0.0036187, 0.0352805, -0.0036187, 0.0352805, -0.0388992, 0.0388992
6: -0.0458881, -0.0068516, -0.0458881, -0.0068516, -0.0390365, 0.0390365
7: 0.8668540, 0.9693018, 0.8668540, 0.9693018, -0.1024479, 0.1024479
8: -0.0103258, 0.0393186, -0.0103258, 0.0393186, -0.0496444, 0.0496444
9: -0.0195926, 0.0167844, -0.0195926, 0.0167844, -0.0363769, 0.0363769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0604432, upper bound: 0.0623941
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0678074, upper bound: 0.0678150
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0167887, 0.0109225, -0.0168542, 0.0108664, -0.0276551, 0.0277767
1: -0.0262801, 0.0082108, -0.0261485, 0.0081316, -0.0344117, 0.0343593
2: 0.0307258, 0.0625459, 0.0308073, 0.0624492, -0.0317234, 0.0317386
3: -0.0010559, 0.0502936, -0.0011743, 0.0501682, -0.0441568, 0.0444450
4: -0.0200746, 0.0180035, -0.0199573, 0.0178441, -0.0379187, 0.0379607
5: -0.0028785, 0.0344613, -0.0027478, 0.0343168, -0.0371953, 0.0372092
6: -0.0451746, -0.0067579, -0.0450486, -0.0066572, -0.0385174, 0.0382907
7: 0.8680803, 0.9694820, 0.8682969, 0.9696760, -0.1015957, 0.1011851
8: -0.0104374, 0.0384653, -0.0105573, 0.0383147, -0.0487521, 0.0490226
9: -0.0192768, 0.0154242, -0.0192211, 0.0151841, -0.0344609, 0.0346453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0540921, upper bound: 0.0537927
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0644251, upper bound: 0.0634923
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0672004, upper bound: 0.0673738
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0167277, 0.0112405, -0.0169960, 0.0104993, -0.0272270, 0.0282365
1: -0.0270252, 0.0086591, -0.0252882, 0.0076141, -0.0346393, 0.0339473
2: 0.0302643, 0.0630935, 0.0313402, 0.0618170, -0.0315527, 0.0317533
3: -0.0009459, 0.0510042, -0.0014304, 0.0493480, -0.0432778, 0.0452653
4: -0.0207392, 0.0189063, -0.0191899, 0.0168018, -0.0375410, 0.0380962
5: -0.0036187, 0.0352805, -0.0018933, 0.0333711, -0.0369898, 0.0371738
6: -0.0458881, -0.0068516, -0.0442249, -0.0064392, -0.0394490, 0.0373733
7: 0.8668540, 0.9693018, 0.8697128, 0.9700955, -0.1032416, 0.0995890
8: -0.0103258, 0.0393186, -0.0108167, 0.0373297, -0.0476556, 0.0501353
9: -0.0195926, 0.0167844, -0.0188566, 0.0136138, -0.0332064, 0.0356409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0592281, upper bound: 0.0553748
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0677004, upper bound: 0.0670149
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0167277, 0.0112405, -0.0169158, 0.0107942, -0.0275219, 0.0281563
1: -0.0270252, 0.0086591, -0.0259793, 0.0080298, -0.0350551, 0.0346384
2: 0.0302643, 0.0630935, 0.0309121, 0.0623249, -0.0320606, 0.0321814
3: -0.0009459, 0.0510042, -0.0012856, 0.0500069, -0.0439162, 0.0451893
4: -0.0207392, 0.0189063, -0.0198064, 0.0176392, -0.0383784, 0.0387127
5: -0.0036187, 0.0352805, -0.0025798, 0.0341308, -0.0377495, 0.0378603
6: -0.0458881, -0.0068516, -0.0448867, -0.0065625, -0.0393257, 0.0380350
7: 0.8668540, 0.9693018, 0.8685753, 0.9698582, -0.1030042, 0.1007265
8: -0.0103258, 0.0393186, -0.0106700, 0.0381210, -0.0484469, 0.0499886
9: -0.0195926, 0.0167844, -0.0191494, 0.0148753, -0.0344679, 0.0359337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0592281, upper bound: 0.0602464
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0677004, upper bound: 0.0674919
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0169960, 0.0104993, -0.0166705, 0.0113047, -0.0283006, 0.0271698
1: -0.0252882, 0.0076141, -0.0271755, 0.0087495, -0.0340377, 0.0347896
2: 0.0313402, 0.0618170, 0.0301712, 0.0632039, -0.0318638, 0.0316459
3: -0.0014304, 0.0493480, -0.0008426, 0.0511474, -0.0454429, 0.0432139
4: -0.0191899, 0.0168018, -0.0208732, 0.0190884, -0.0382783, 0.0376751
5: -0.0018933, 0.0333711, -0.0037680, 0.0354457, -0.0373390, 0.0371391
6: -0.0442249, -0.0064392, -0.0460320, -0.0069396, -0.0372853, 0.0395928
7: 0.8697128, 0.9700955, 0.8666067, 0.9691324, -0.0994195, 0.1034889
8: -0.0108167, 0.0373297, -0.0102211, 0.0394907, -0.0503074, 0.0475509
9: -0.0188566, 0.0136138, -0.0196563, 0.0170587, -0.0359152, 0.0332701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0529856, upper bound: 0.0544263
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0630477, upper bound: 0.0640998
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0669910, upper bound: 0.0676743
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0169158, 0.0107942, -0.0167887, 0.0109225, -0.0278383, 0.0275829
1: -0.0259793, 0.0080298, -0.0262801, 0.0082108, -0.0341901, 0.0343099
2: 0.0309121, 0.0623249, 0.0307258, 0.0625459, -0.0316338, 0.0315991
3: -0.0012856, 0.0500069, -0.0010559, 0.0502936, -0.0445244, 0.0439625
4: -0.0198064, 0.0176392, -0.0200746, 0.0180035, -0.0378098, 0.0377138
5: -0.0025798, 0.0341308, -0.0028785, 0.0344613, -0.0370412, 0.0370093
6: -0.0448867, -0.0065625, -0.0451746, -0.0067579, -0.0381287, 0.0386121
7: 0.8685753, 0.9698582, 0.8680803, 0.9694820, -0.1009067, 0.1017779
8: -0.0106700, 0.0381210, -0.0104374, 0.0384653, -0.0491353, 0.0485584
9: -0.0191494, 0.0148753, -0.0192768, 0.0154242, -0.0345736, 0.0341521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0550479, upper bound: 0.0541186
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673998, upper bound: 0.0672252
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0169158, 0.0107942, -0.0167277, 0.0112405, -0.0281563, 0.0275219
1: -0.0259793, 0.0080298, -0.0270252, 0.0086591, -0.0346384, 0.0350551
2: 0.0309121, 0.0623249, 0.0302643, 0.0630935, -0.0321814, 0.0320606
3: -0.0012856, 0.0500069, -0.0009459, 0.0510042, -0.0451893, 0.0439162
4: -0.0198064, 0.0176392, -0.0207392, 0.0189063, -0.0387127, 0.0383784
5: -0.0025798, 0.0341308, -0.0036187, 0.0352805, -0.0378603, 0.0377495
6: -0.0448867, -0.0065625, -0.0458881, -0.0068516, -0.0380350, 0.0393257
7: 0.8685753, 0.9698582, 0.8668540, 0.9693018, -0.1007265, 0.1030042
8: -0.0106700, 0.0381210, -0.0103258, 0.0393186, -0.0499886, 0.0484469
9: -0.0191494, 0.0148753, -0.0195926, 0.0167844, -0.0359337, 0.0344679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0550479, upper bound: 0.0601044
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673998, upper bound: 0.0676593
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0169960, 0.0104993, -0.0168542, 0.0108664, -0.0278624, 0.0273535
1: -0.0252882, 0.0076141, -0.0261485, 0.0081316, -0.0334199, 0.0337625
2: 0.0313402, 0.0618170, 0.0308073, 0.0624492, -0.0311090, 0.0310097
3: -0.0014304, 0.0493480, -0.0011743, 0.0501682, -0.0445056, 0.0435372
4: -0.0191899, 0.0168018, -0.0199573, 0.0178441, -0.0370340, 0.0367591
5: -0.0018933, 0.0333711, -0.0027478, 0.0343168, -0.0362100, 0.0361189
6: -0.0442249, -0.0064392, -0.0450486, -0.0066572, -0.0375677, 0.0386094
7: 0.8697128, 0.9700955, 0.8682969, 0.9696760, -0.0999632, 0.1017986
8: -0.0108167, 0.0373297, -0.0105573, 0.0383147, -0.0491314, 0.0478870
9: -0.0188566, 0.0136138, -0.0192211, 0.0151841, -0.0340406, 0.0328349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0529856, upper bound: 0.0537658
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0630475, upper bound: 0.0632665
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0669910, upper bound: 0.0673804
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0169158, 0.0107942, -0.0169960, 0.0104993, -0.0274151, 0.0277902
1: -0.0259793, 0.0080298, -0.0252882, 0.0076141, -0.0335934, 0.0333181
2: 0.0309121, 0.0623249, 0.0313402, 0.0618170, -0.0309049, 0.0309848
3: -0.0012856, 0.0500069, -0.0014304, 0.0493480, -0.0436045, 0.0443099
4: -0.0198064, 0.0176392, -0.0191899, 0.0168018, -0.0366082, 0.0368291
5: -0.0025798, 0.0341308, -0.0018933, 0.0333711, -0.0359509, 0.0360241
6: -0.0448867, -0.0065625, -0.0442249, -0.0064392, -0.0384475, 0.0376625
7: 0.8685753, 0.9698582, 0.8697128, 0.9700955, -0.1015202, 0.1001453
8: -0.0106700, 0.0381210, -0.0108167, 0.0373297, -0.0479998, 0.0489377
9: -0.0191494, 0.0148753, -0.0188566, 0.0136138, -0.0327632, 0.0337319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0550262, upper bound: 0.0533913
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673998, upper bound: 0.0670157
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0169158, 0.0107942, -0.0169158, 0.0107942, -0.0277100, 0.0277100
1: -0.0259793, 0.0080298, -0.0259793, 0.0080298, -0.0340092, 0.0340092
2: 0.0309121, 0.0623249, 0.0309121, 0.0623249, -0.0314128, 0.0314128
3: -0.0012856, 0.0500069, -0.0012856, 0.0500069, -0.0442472, 0.0442472
4: -0.0198064, 0.0176392, -0.0198064, 0.0176392, -0.0374455, 0.0374455
5: -0.0025798, 0.0341308, -0.0025798, 0.0341308, -0.0367106, 0.0367106
6: -0.0448867, -0.0065625, -0.0448867, -0.0065625, -0.0383242, 0.0383242
7: 0.8685753, 0.9698582, 0.8685753, 0.9698582, -0.1012828, 0.1012828
8: -0.0106700, 0.0381210, -0.0106700, 0.0381210, -0.0487910, 0.0487910
9: -0.0191494, 0.0148753, -0.0191494, 0.0148753, -0.0340247, 0.0340247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0550262, upper bound: 0.0595155
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673998, upper bound: 0.0674919
time: 0.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.88 seconds
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0652719, upper bound: 0.0654074
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0673213, upper bound: 0.0677832
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0604432, upper bound: 0.0577679
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0678074, upper bound: 0.0673449
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0604432, upper bound: 0.0623941
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0678074, upper bound: 0.0678150
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0644251, upper bound: 0.0634923
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0672004, upper bound: 0.0673738
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0592281, upper bound: 0.0553748
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0677004, upper bound: 0.0670149
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0592281, upper bound: 0.0602464
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0677004, upper bound: 0.0674919
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0630477, upper bound: 0.0640998
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0669910, upper bound: 0.0676743
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0550479, upper bound: 0.0541186
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0673998, upper bound: 0.0672252
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0550479, upper bound: 0.0601044
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0673998, upper bound: 0.0676593
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0630475, upper bound: 0.0632665
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0669910, upper bound: 0.0673804
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0550262, upper bound: 0.0533913
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0673998, upper bound: 0.0670157
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0550262, upper bound: 0.0595155
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 7, lower bound: -0.0673998, upper bound: 0.0674919

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0167800, 0.0109073, -0.0166705, 0.0113047, -0.0280847, 0.0275777
1: -0.0262443, 0.0081892, -0.0271755, 0.0087495, -0.0349938, 0.0353648
2: 0.0307480, 0.0625196, 0.0301712, 0.0632039, -0.0324559, 0.0323484
3: -0.0010405, 0.0502595, -0.0008426, 0.0511474, -0.0449959, 0.0440184
4: -0.0200426, 0.0179601, -0.0208732, 0.0190884, -0.0391310, 0.0388333
5: -0.0028429, 0.0344220, -0.0037680, 0.0354457, -0.0382887, 0.0381899
6: -0.0451403, -0.0067712, -0.0460320, -0.0069396, -0.0382007, 0.0392609
7: 0.8681393, 0.9694566, 0.8666067, 0.9691324, -0.1009930, 0.1028499
8: -0.0104216, 0.0384243, -0.0102211, 0.0394907, -0.0499123, 0.0486455
9: -0.0192617, 0.0153589, -0.0196563, 0.0170587, -0.0363203, 0.0350152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673213, upper bound: 0.0673319
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673213, upper bound: 0.0677832
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0166697, 0.0112259, -0.0167887, 0.0109225, -0.0275922, 0.0280145
1: -0.0269909, 0.0086384, -0.0262801, 0.0082108, -0.0352017, 0.0349185
2: 0.0302856, 0.0630682, 0.0307258, 0.0625459, -0.0322603, 0.0323424
3: -0.0008411, 0.0509714, -0.0010559, 0.0502936, -0.0439788, 0.0448288
4: -0.0207086, 0.0188646, -0.0200746, 0.0180035, -0.0387120, 0.0389392
5: -0.0035845, 0.0352427, -0.0028785, 0.0344613, -0.0380459, 0.0381212
6: -0.0458552, -0.0069408, -0.0451746, -0.0067579, -0.0390972, 0.0382337
7: 0.8669106, 0.9691300, 0.8680803, 0.9694820, -0.1025714, 0.1010497
8: -0.0102197, 0.0392792, -0.0104374, 0.0384653, -0.0486850, 0.0497165
9: -0.0195780, 0.0167216, -0.0192768, 0.0154242, -0.0350022, 0.0359984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0545572, upper bound: 0.0540954
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0654074, upper bound: 0.0652719
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0677832, upper bound: 0.0673213
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0166697, 0.0112259, -0.0167277, 0.0112405, -0.0279102, 0.0279536
1: -0.0269909, 0.0086384, -0.0270252, 0.0086591, -0.0356499, 0.0356637
2: 0.0302856, 0.0630682, 0.0302643, 0.0630935, -0.0328078, 0.0328039
3: -0.0008411, 0.0509714, -0.0009459, 0.0510042, -0.0446965, 0.0447902
4: -0.0207086, 0.0188646, -0.0207392, 0.0189063, -0.0396149, 0.0396038
5: -0.0035845, 0.0352427, -0.0036187, 0.0352805, -0.0388650, 0.0388614
6: -0.0458552, -0.0069408, -0.0458881, -0.0068516, -0.0390036, 0.0389473
7: 0.8669106, 0.9691300, 0.8668540, 0.9693018, -0.1023912, 0.1022761
8: -0.0102197, 0.0392792, -0.0103258, 0.0393186, -0.0495383, 0.0496050
9: -0.0195780, 0.0167216, -0.0195926, 0.0167844, -0.0363624, 0.0363142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0625710, upper bound: 0.0634552
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0625710, upper bound: 0.0678150
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0167800, 0.0109073, -0.0168542, 0.0108664, -0.0276464, 0.0277614
1: -0.0262443, 0.0081892, -0.0261485, 0.0081316, -0.0343759, 0.0343377
2: 0.0307480, 0.0625196, 0.0308073, 0.0624492, -0.0317012, 0.0317123
3: -0.0010405, 0.0502595, -0.0011743, 0.0501682, -0.0441158, 0.0443808
4: -0.0200426, 0.0179601, -0.0199573, 0.0178441, -0.0378867, 0.0379174
5: -0.0028429, 0.0344220, -0.0027478, 0.0343168, -0.0371597, 0.0371698
6: -0.0451403, -0.0067712, -0.0450486, -0.0066572, -0.0384831, 0.0382774
7: 0.8681393, 0.9694566, 0.8682969, 0.9696760, -0.1015367, 0.1011596
8: -0.0104216, 0.0384243, -0.0105573, 0.0383147, -0.0487363, 0.0489816
9: -0.0192617, 0.0153589, -0.0192211, 0.0151841, -0.0344457, 0.0345800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0672004, upper bound: 0.0669975
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0672004, upper bound: 0.0673738
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0166697, 0.0112259, -0.0169960, 0.0104993, -0.0271690, 0.0282218
1: -0.0269909, 0.0086384, -0.0252882, 0.0076141, -0.0346049, 0.0339267
2: 0.0302856, 0.0630682, 0.0313402, 0.0618170, -0.0315314, 0.0317281
3: -0.0008411, 0.0509714, -0.0014304, 0.0493480, -0.0431144, 0.0452347
4: -0.0207086, 0.0188646, -0.0191899, 0.0168018, -0.0375104, 0.0380546
5: -0.0035845, 0.0352427, -0.0018933, 0.0333711, -0.0369556, 0.0371360
6: -0.0458552, -0.0069408, -0.0442249, -0.0064392, -0.0394160, 0.0372841
7: 0.8669106, 0.9691300, 0.8697128, 0.9700955, -0.1031849, 0.0994172
8: -0.0102197, 0.0392792, -0.0108167, 0.0373297, -0.0475494, 0.0500959
9: -0.0195780, 0.0167216, -0.0188566, 0.0136138, -0.0331918, 0.0355782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0544263, upper bound: 0.0529856
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0640998, upper bound: 0.0630477
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0676743, upper bound: 0.0669910
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0166697, 0.0112259, -0.0169158, 0.0107942, -0.0274639, 0.0281416
1: -0.0269909, 0.0086384, -0.0259793, 0.0080298, -0.0350207, 0.0346178
2: 0.0302856, 0.0630682, 0.0309121, 0.0623249, -0.0320393, 0.0321561
3: -0.0008411, 0.0509714, -0.0012856, 0.0500069, -0.0437966, 0.0451590
4: -0.0207086, 0.0188646, -0.0198064, 0.0176392, -0.0383477, 0.0386710
5: -0.0035845, 0.0352427, -0.0025798, 0.0341308, -0.0377153, 0.0378225
6: -0.0458552, -0.0069408, -0.0448867, -0.0065625, -0.0392927, 0.0379458
7: 0.8669106, 0.9691300, 0.8685753, 0.9698582, -0.1029475, 0.1005547
8: -0.0102197, 0.0392792, -0.0106700, 0.0381210, -0.0483407, 0.0499492
9: -0.0195780, 0.0167216, -0.0191494, 0.0148753, -0.0344533, 0.0358710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0601961, upper bound: 0.0584508
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0601961, upper bound: 0.0674919
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0169868, 0.0104848, -0.0166705, 0.0113047, -0.0282915, 0.0271553
1: -0.0252545, 0.0075937, -0.0271755, 0.0087495, -0.0340040, 0.0347693
2: 0.0313611, 0.0617922, 0.0301712, 0.0632039, -0.0318429, 0.0316210
3: -0.0014139, 0.0493159, -0.0008426, 0.0511474, -0.0454005, 0.0431484
4: -0.0191598, 0.0167609, -0.0208732, 0.0190884, -0.0382482, 0.0376342
5: -0.0018597, 0.0333340, -0.0037680, 0.0354457, -0.0373055, 0.0371020
6: -0.0441926, -0.0064532, -0.0460320, -0.0069396, -0.0372529, 0.0395788
7: 0.8697683, 0.9700685, 0.8666067, 0.9691324, -0.0993640, 0.1034618
8: -0.0108000, 0.0372911, -0.0102211, 0.0394907, -0.0502907, 0.0475122
9: -0.0188423, 0.0135521, -0.0196563, 0.0170587, -0.0359009, 0.0332083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0669910, upper bound: 0.0672123
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0669910, upper bound: 0.0676743
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0168535, 0.0107802, -0.0167887, 0.0109225, -0.0277760, 0.0275689
1: -0.0259466, 0.0080101, -0.0262801, 0.0082108, -0.0341574, 0.0342902
2: 0.0309324, 0.0623009, 0.0307258, 0.0625459, -0.0316135, 0.0315751
3: -0.0011731, 0.0499757, -0.0010559, 0.0502936, -0.0443491, 0.0439330
4: -0.0197772, 0.0175995, -0.0200746, 0.0180035, -0.0377806, 0.0376741
5: -0.0025473, 0.0340948, -0.0028785, 0.0344613, -0.0370086, 0.0369733
6: -0.0448553, -0.0066582, -0.0451746, -0.0067579, -0.0380974, 0.0385164
7: 0.8686290, 0.9696742, 0.8680803, 0.9694820, -0.1008530, 0.1015939
8: -0.0105561, 0.0380835, -0.0104374, 0.0384653, -0.0490214, 0.0485209
9: -0.0191355, 0.0148156, -0.0192768, 0.0154242, -0.0345597, 0.0340924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0537927, upper bound: 0.0540921
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0634923, upper bound: 0.0644251
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673738, upper bound: 0.0672004
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0168535, 0.0107802, -0.0167277, 0.0112405, -0.0280940, 0.0275080
1: -0.0259466, 0.0080101, -0.0270252, 0.0086591, -0.0346057, 0.0350354
2: 0.0309324, 0.0623009, 0.0302643, 0.0630935, -0.0321611, 0.0320366
3: -0.0011731, 0.0499757, -0.0009459, 0.0510042, -0.0450638, 0.0438868
4: -0.0197772, 0.0175995, -0.0207392, 0.0189063, -0.0386835, 0.0383387
5: -0.0025473, 0.0340948, -0.0036187, 0.0352805, -0.0378278, 0.0377135
6: -0.0448553, -0.0066582, -0.0458881, -0.0068516, -0.0380037, 0.0392300
7: 0.8686290, 0.9696742, 0.8668540, 0.9693018, -0.1006728, 0.1028202
8: -0.0105561, 0.0380835, -0.0103258, 0.0393186, -0.0498747, 0.0484094
9: -0.0191355, 0.0148156, -0.0195926, 0.0167844, -0.0359199, 0.0344082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0603109, upper bound: 0.0624198
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0603109, upper bound: 0.0676593
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0169868, 0.0104848, -0.0168542, 0.0108664, -0.0278532, 0.0273390
1: -0.0252545, 0.0075937, -0.0261485, 0.0081316, -0.0333861, 0.0337422
2: 0.0313611, 0.0617922, 0.0308073, 0.0624492, -0.0310881, 0.0309849
3: -0.0014139, 0.0493159, -0.0011743, 0.0501682, -0.0444669, 0.0434795
4: -0.0191598, 0.0167609, -0.0199573, 0.0178441, -0.0370039, 0.0367182
5: -0.0018597, 0.0333340, -0.0027478, 0.0343168, -0.0361765, 0.0360818
6: -0.0441926, -0.0064532, -0.0450486, -0.0066572, -0.0375354, 0.0385953
7: 0.8697683, 0.9700685, 0.8682969, 0.9696760, -0.0999077, 0.1017715
8: -0.0108000, 0.0372911, -0.0105573, 0.0383147, -0.0491147, 0.0478483
9: -0.0188423, 0.0135521, -0.0192211, 0.0151841, -0.0340263, 0.0327731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0669910, upper bound: 0.0669975
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0669910, upper bound: 0.0673804
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0168535, 0.0107802, -0.0169960, 0.0104993, -0.0273528, 0.0277762
1: -0.0259466, 0.0080101, -0.0252882, 0.0076141, -0.0335607, 0.0332984
2: 0.0309324, 0.0623009, 0.0313402, 0.0618170, -0.0308847, 0.0309607
3: -0.0011731, 0.0499757, -0.0014304, 0.0493480, -0.0434456, 0.0442803
4: -0.0197772, 0.0175995, -0.0191899, 0.0168018, -0.0365790, 0.0367895
5: -0.0025473, 0.0340948, -0.0018933, 0.0333711, -0.0359184, 0.0359881
6: -0.0448553, -0.0066582, -0.0442249, -0.0064392, -0.0384161, 0.0375667
7: 0.8686290, 0.9696742, 0.8697128, 0.9700955, -0.1014665, 0.0999613
8: -0.0105561, 0.0380835, -0.0108167, 0.0373297, -0.0478858, 0.0489002
9: -0.0191355, 0.0148156, -0.0188566, 0.0136138, -0.0327493, 0.0336721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0537658, upper bound: 0.0530896
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0632665, upper bound: 0.0630475
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673738, upper bound: 0.0669917
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0168535, 0.0107802, -0.0169158, 0.0107942, -0.0276477, 0.0276960
1: -0.0259466, 0.0080101, -0.0259793, 0.0080298, -0.0339764, 0.0339895
2: 0.0309324, 0.0623009, 0.0309121, 0.0623249, -0.0313925, 0.0313888
3: -0.0011731, 0.0499757, -0.0012856, 0.0500069, -0.0441204, 0.0442179
4: -0.0197772, 0.0175995, -0.0198064, 0.0176392, -0.0374163, 0.0374059
5: -0.0025473, 0.0340948, -0.0025798, 0.0341308, -0.0366781, 0.0366746
6: -0.0448553, -0.0066582, -0.0448867, -0.0065625, -0.0382928, 0.0382285
7: 0.8686290, 0.9696742, 0.8685753, 0.9698582, -0.1012292, 0.1010988
8: -0.0105561, 0.0380835, -0.0106700, 0.0381210, -0.0486771, 0.0487535
9: -0.0191355, 0.0148156, -0.0191494, 0.0148753, -0.0340108, 0.0339650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0595430, upper bound: 0.0583411
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0595430, upper bound: 0.0674919
time: 0.61 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.83 seconds
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0673213, upper bound: 0.0673319
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0673213, upper bound: 0.0677832
IS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0654074, upper bound: 0.0652719
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0677832, upper bound: 0.0673213
IS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0625710, upper bound: 0.0634552
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0625710, upper bound: 0.0678150
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0672004, upper bound: 0.0669975
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0672004, upper bound: 0.0673738
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0640998, upper bound: 0.0630477
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0676743, upper bound: 0.0669910
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0601961, upper bound: 0.0584508
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0601961, upper bound: 0.0674919
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0669910, upper bound: 0.0672123
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0669910, upper bound: 0.0676743
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0634923, upper bound: 0.0644251
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0673738, upper bound: 0.0672004
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0603109, upper bound: 0.0624198
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0603109, upper bound: 0.0676593
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0669910, upper bound: 0.0669975
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0669910, upper bound: 0.0673804
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0632665, upper bound: 0.0630475
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0673738, upper bound: 0.0669917
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0595430, upper bound: 0.0583411
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 7, lower bound: -0.0595430, upper bound: 0.0674919

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0167800, 0.0109073, -0.0167290, 0.0109086, -0.0276886, 0.0276363
1: -0.0262443, 0.0081892, -0.0262473, 0.0081911, -0.0344353, 0.0344365
2: 0.0307480, 0.0625196, 0.0307461, 0.0625218, -0.0317738, 0.0317735
3: -0.0010405, 0.0502595, -0.0009482, 0.0502625, -0.0441629, 0.0440046
4: -0.0200426, 0.0179601, -0.0200454, 0.0179638, -0.0380064, 0.0380055
5: -0.0028429, 0.0344220, -0.0028460, 0.0344253, -0.0372683, 0.0372680
6: -0.0451403, -0.0067712, -0.0451432, -0.0068496, -0.0382906, 0.0383720
7: 0.8681393, 0.9694566, 0.8681344, 0.9693056, -0.1011663, 0.1013222
8: -0.0104216, 0.0384243, -0.0103282, 0.0384278, -0.0488495, 0.0487525
9: -0.0192617, 0.0153589, -0.0192629, 0.0153644, -0.0346261, 0.0346218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 245
type: A, layer: 3, pos: 190
type: A, layer: 3, pos: 249
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 103
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 129

Time for candidate selection: 4.11 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0552086, upper bound: 0.0658428
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0661403, upper bound: 0.0661415
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0167800, 0.0109073, -0.0166697, 0.0112259, -0.0280059, 0.0275769
1: -0.0262443, 0.0081892, -0.0269909, 0.0086384, -0.0348827, 0.0351801
2: 0.0307480, 0.0625196, 0.0302856, 0.0630682, -0.0323202, 0.0322340
3: -0.0010405, 0.0502595, -0.0008411, 0.0509714, -0.0447877, 0.0439185
4: -0.0200426, 0.0179601, -0.0207086, 0.0188646, -0.0389073, 0.0386687
5: -0.0028429, 0.0344220, -0.0035845, 0.0352427, -0.0380856, 0.0380065
6: -0.0451403, -0.0067712, -0.0458552, -0.0069408, -0.0381994, 0.0390840
7: 0.8681393, 0.9694566, 0.8669106, 0.9691300, -0.1009907, 0.1025459
8: -0.0104216, 0.0384243, -0.0102197, 0.0392792, -0.0497008, 0.0486440
9: -0.0192617, 0.0153589, -0.0195780, 0.0167216, -0.0359833, 0.0349369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 245
type: A, layer: 3, pos: 190
type: A, layer: 3, pos: 249
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 103
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 129

Time for candidate selection: 4.13 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0552086, upper bound: 0.0661259
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0661403, upper bound: 0.0665076
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0166697, 0.0112259, -0.0167800, 0.0109073, -0.0275769, 0.0280059
1: -0.0269909, 0.0086384, -0.0262443, 0.0081892, -0.0351801, 0.0348827
2: 0.0302856, 0.0630682, 0.0307480, 0.0625196, -0.0322340, 0.0323202
3: -0.0008411, 0.0509714, -0.0010405, 0.0502595, -0.0439185, 0.0447877
4: -0.0207086, 0.0188646, -0.0200426, 0.0179601, -0.0386687, 0.0389073
5: -0.0035845, 0.0352427, -0.0028429, 0.0344220, -0.0380065, 0.0380856
6: -0.0458552, -0.0069408, -0.0451403, -0.0067712, -0.0390840, 0.0381994
7: 0.8669106, 0.9691300, 0.8681393, 0.9694566, -0.1025459, 0.1009907
8: -0.0102197, 0.0392792, -0.0104216, 0.0384243, -0.0486440, 0.0497008
9: -0.0195780, 0.0167216, -0.0192617, 0.0153589, -0.0349369, 0.0359833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0663063, upper bound: 0.0647610
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0663063, upper bound: 0.0673213
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0166697, 0.0112259, -0.0166697, 0.0112259, -0.0278955, 0.0278955
1: -0.0269909, 0.0086384, -0.0269909, 0.0086384, -0.0356293, 0.0356293
2: 0.0302856, 0.0630682, 0.0302856, 0.0630682, -0.0327826, 0.0327826
3: -0.0008411, 0.0509714, -0.0008411, 0.0509714, -0.0446656, 0.0446656
4: -0.0207086, 0.0188646, -0.0207086, 0.0188646, -0.0395732, 0.0395732
5: -0.0035845, 0.0352427, -0.0035845, 0.0352427, -0.0388272, 0.0388272
6: -0.0458552, -0.0069408, -0.0458552, -0.0069408, -0.0389143, 0.0389143
7: 0.8669106, 0.9691300, 0.8669106, 0.9691300, -0.1022194, 0.1022194
8: -0.0102197, 0.0392792, -0.0102197, 0.0392792, -0.0494989, 0.0494989
9: -0.0195780, 0.0167216, -0.0195780, 0.0167216, -0.0362996, 0.0362996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0566410, upper bound: 0.0664480
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0625574, upper bound: 0.0677959
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0167800, 0.0109073, -0.0169339, 0.0104859, -0.0272659, 0.0278411
1: -0.0262443, 0.0081892, -0.0252569, 0.0075952, -0.0338394, 0.0334461
2: 0.0307480, 0.0625196, 0.0313596, 0.0617940, -0.0310460, 0.0311600
3: -0.0010405, 0.0502595, -0.0013182, 0.0493181, -0.0432890, 0.0443975
4: -0.0200426, 0.0179601, -0.0191620, 0.0167639, -0.0368065, 0.0371221
5: -0.0028429, 0.0344220, -0.0018622, 0.0333366, -0.0361795, 0.0362842
6: -0.0451403, -0.0067712, -0.0441948, -0.0065347, -0.0386056, 0.0374237
7: 0.8681393, 0.9694566, 0.8697644, 0.9699117, -0.1017724, 0.0996921
8: -0.0104216, 0.0384243, -0.0107031, 0.0372937, -0.0477154, 0.0491274
9: -0.0192617, 0.0153589, -0.0188432, 0.0135566, -0.0328182, 0.0342022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 245
type: A, layer: 3, pos: 190
type: A, layer: 3, pos: 249
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 103
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 129

Time for candidate selection: 4.19 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0545920, upper bound: 0.0655103
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0659800, upper bound: 0.0657268
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0167800, 0.0109073, -0.0168535, 0.0107802, -0.0275603, 0.0277608
1: -0.0262443, 0.0081892, -0.0259466, 0.0080101, -0.0342544, 0.0341358
2: 0.0307480, 0.0625196, 0.0309324, 0.0623009, -0.0315529, 0.0315872
3: -0.0010405, 0.0502595, -0.0011731, 0.0499757, -0.0438920, 0.0442855
4: -0.0200426, 0.0179601, -0.0197772, 0.0175995, -0.0376421, 0.0377373
5: -0.0028429, 0.0344220, -0.0025473, 0.0340948, -0.0369377, 0.0369693
6: -0.0451403, -0.0067712, -0.0448553, -0.0066582, -0.0384821, 0.0380841
7: 0.8681393, 0.9694566, 0.8686290, 0.9696742, -0.1015348, 0.1008276
8: -0.0104216, 0.0384243, -0.0105561, 0.0380835, -0.0485052, 0.0489804
9: -0.0192617, 0.0153589, -0.0191355, 0.0148156, -0.0340772, 0.0344945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 245
type: A, layer: 3, pos: 190
type: A, layer: 3, pos: 249
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 103
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 129

Time for candidate selection: 4.10 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0545920, upper bound: 0.0657510
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0659800, upper bound: 0.0660391
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0166697, 0.0112259, -0.0169868, 0.0104848, -0.0271545, 0.0282127
1: -0.0269909, 0.0086384, -0.0252545, 0.0075937, -0.0345846, 0.0338929
2: 0.0302856, 0.0630682, 0.0313611, 0.0617922, -0.0315066, 0.0317072
3: -0.0008411, 0.0509714, -0.0014139, 0.0493159, -0.0430484, 0.0451923
4: -0.0207086, 0.0188646, -0.0191598, 0.0167609, -0.0374695, 0.0380244
5: -0.0035845, 0.0352427, -0.0018597, 0.0333340, -0.0369185, 0.0371024
6: -0.0458552, -0.0069408, -0.0441926, -0.0064532, -0.0394019, 0.0372517
7: 0.8669106, 0.9691300, 0.8697683, 0.9700685, -0.1031578, 0.0993617
8: -0.0102197, 0.0392792, -0.0108000, 0.0372911, -0.0475107, 0.0500792
9: -0.0195780, 0.0167216, -0.0188423, 0.0135521, -0.0331301, 0.0355639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0656925, upper bound: 0.0628824
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0656925, upper bound: 0.0669910
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0166697, 0.0112259, -0.0168535, 0.0107802, -0.0274499, 0.0280794
1: -0.0269909, 0.0086384, -0.0259466, 0.0080101, -0.0350010, 0.0345850
2: 0.0302856, 0.0630682, 0.0309324, 0.0623009, -0.0320153, 0.0321358
3: -0.0008411, 0.0509714, -0.0011731, 0.0499757, -0.0437673, 0.0450329
4: -0.0207086, 0.0188646, -0.0197772, 0.0175995, -0.0383081, 0.0386418
5: -0.0035845, 0.0352427, -0.0025473, 0.0340948, -0.0376793, 0.0377900
6: -0.0458552, -0.0069408, -0.0448553, -0.0066582, -0.0391970, 0.0379144
7: 0.8669106, 0.9691300, 0.8686290, 0.9696742, -0.1027635, 0.1005011
8: -0.0102197, 0.0392792, -0.0105561, 0.0380835, -0.0483032, 0.0498353
9: -0.0195780, 0.0167216, -0.0191355, 0.0148156, -0.0343936, 0.0358571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0526818, upper bound: 0.0650821
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0601608, upper bound: 0.0674736
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0169868, 0.0104848, -0.0167290, 0.0109086, -0.0278954, 0.0272138
1: -0.0252545, 0.0075937, -0.0262473, 0.0081911, -0.0334455, 0.0338411
2: 0.0313611, 0.0617922, 0.0307461, 0.0625218, -0.0311608, 0.0310461
3: -0.0014139, 0.0493159, -0.0009482, 0.0502625, -0.0445675, 0.0431354
4: -0.0191598, 0.0167609, -0.0200454, 0.0179638, -0.0371236, 0.0368063
5: -0.0018597, 0.0333340, -0.0028460, 0.0344253, -0.0362851, 0.0361800
6: -0.0441926, -0.0064532, -0.0451432, -0.0068496, -0.0373429, 0.0386899
7: 0.8697683, 0.9700685, 0.8681344, 0.9693056, -0.0995373, 0.1019341
8: -0.0108000, 0.0372911, -0.0103282, 0.0384278, -0.0492278, 0.0476193
9: -0.0188423, 0.0135521, -0.0192629, 0.0153644, -0.0342067, 0.0328150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 245
type: A, layer: 3, pos: 190
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 249
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 103
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 129

Time for candidate selection: 4.19 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0532431, upper bound: 0.0656988
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0657268, upper bound: 0.0659828
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0169868, 0.0104848, -0.0166697, 0.0112259, -0.0282127, 0.0271545
1: -0.0252545, 0.0075937, -0.0269909, 0.0086384, -0.0338929, 0.0345846
2: 0.0313611, 0.0617922, 0.0302856, 0.0630682, -0.0317072, 0.0315066
3: -0.0014139, 0.0493159, -0.0008411, 0.0509714, -0.0451924, 0.0430484
4: -0.0191598, 0.0167609, -0.0207086, 0.0188646, -0.0380244, 0.0374695
5: -0.0018597, 0.0333340, -0.0035845, 0.0352427, -0.0371024, 0.0369185
6: -0.0441926, -0.0064532, -0.0458552, -0.0069408, -0.0372517, 0.0394019
7: 0.8697683, 0.9700685, 0.8669106, 0.9691300, -0.0993617, 0.1031578
8: -0.0108000, 0.0372911, -0.0102197, 0.0392792, -0.0500792, 0.0475107
9: -0.0188423, 0.0135521, -0.0195780, 0.0167216, -0.0355639, 0.0331301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 245
type: A, layer: 3, pos: 190
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 249
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 103
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 129

Time for candidate selection: 4.10 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0532431, upper bound: 0.0659990
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0657268, upper bound: 0.0663497
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0168535, 0.0107802, -0.0167800, 0.0109073, -0.0277608, 0.0275603
1: -0.0259466, 0.0080101, -0.0262443, 0.0081892, -0.0341358, 0.0342544
2: 0.0309324, 0.0623009, 0.0307480, 0.0625196, -0.0315872, 0.0315529
3: -0.0011731, 0.0499757, -0.0010405, 0.0502595, -0.0442855, 0.0438920
4: -0.0197772, 0.0175995, -0.0200426, 0.0179601, -0.0377373, 0.0376421
5: -0.0025473, 0.0340948, -0.0028429, 0.0344220, -0.0369693, 0.0369377
6: -0.0448553, -0.0066582, -0.0451403, -0.0067712, -0.0380841, 0.0384821
7: 0.8686290, 0.9696742, 0.8681393, 0.9694566, -0.1008276, 0.1015348
8: -0.0105561, 0.0380835, -0.0104216, 0.0384243, -0.0489804, 0.0485052
9: -0.0191355, 0.0148156, -0.0192617, 0.0153589, -0.0344945, 0.0340772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0639169, upper bound: 0.0636421
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0639169, upper bound: 0.0672004
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0168535, 0.0107802, -0.0166697, 0.0112259, -0.0280794, 0.0274499
1: -0.0259466, 0.0080101, -0.0269909, 0.0086384, -0.0345850, 0.0350010
2: 0.0309324, 0.0623009, 0.0302856, 0.0630682, -0.0321358, 0.0320153
3: -0.0011731, 0.0499757, -0.0008411, 0.0509714, -0.0450329, 0.0437673
4: -0.0197772, 0.0175995, -0.0207086, 0.0188646, -0.0386418, 0.0383081
5: -0.0025473, 0.0340948, -0.0035845, 0.0352427, -0.0377900, 0.0376793
6: -0.0448553, -0.0066582, -0.0458552, -0.0069408, -0.0379144, 0.0391970
7: 0.8686290, 0.9696742, 0.8669106, 0.9691300, -0.1005011, 0.1027635
8: -0.0105561, 0.0380835, -0.0102197, 0.0392792, -0.0498353, 0.0483032
9: -0.0191355, 0.0148156, -0.0195780, 0.0167216, -0.0358571, 0.0343936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0529687, upper bound: 0.0655753
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0602970, upper bound: 0.0676396
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0169868, 0.0104848, -0.0169339, 0.0104859, -0.0274727, 0.0274187
1: -0.0252545, 0.0075937, -0.0252569, 0.0075952, -0.0328496, 0.0328506
2: 0.0313611, 0.0617922, 0.0313596, 0.0617940, -0.0304330, 0.0304326
3: -0.0014139, 0.0493159, -0.0013182, 0.0493181, -0.0436407, 0.0434936
4: -0.0191598, 0.0167609, -0.0191620, 0.0167639, -0.0359237, 0.0359229
5: -0.0018597, 0.0333340, -0.0018622, 0.0333366, -0.0351963, 0.0351962
6: -0.0441926, -0.0064532, -0.0441948, -0.0065347, -0.0376579, 0.0377416
7: 0.8697683, 0.9700685, 0.8697644, 0.9699117, -0.1001434, 0.1003040
8: -0.0108000, 0.0372911, -0.0107031, 0.0372937, -0.0480937, 0.0479941
9: -0.0188423, 0.0135521, -0.0188432, 0.0135566, -0.0323988, 0.0323953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 245
type: A, layer: 3, pos: 190
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 249
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 103
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 129

Time for candidate selection: 4.13 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0532431, upper bound: 0.0655135
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0657268, upper bound: 0.0657268
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0169868, 0.0104848, -0.0168535, 0.0107802, -0.0277671, 0.0273384
1: -0.0252545, 0.0075937, -0.0259466, 0.0080101, -0.0332646, 0.0335403
2: 0.0313611, 0.0617922, 0.0309324, 0.0623009, -0.0309398, 0.0308598
3: -0.0014139, 0.0493159, -0.0011731, 0.0499757, -0.0442417, 0.0433868
4: -0.0191598, 0.0167609, -0.0197772, 0.0175995, -0.0367593, 0.0365381
5: -0.0018597, 0.0333340, -0.0025473, 0.0340948, -0.0359545, 0.0358813
6: -0.0441926, -0.0064532, -0.0448553, -0.0066582, -0.0375344, 0.0384020
7: 0.8697683, 0.9700685, 0.8686290, 0.9696742, -0.0999058, 0.1014395
8: -0.0108000, 0.0372911, -0.0105561, 0.0380835, -0.0488835, 0.0478472
9: -0.0188423, 0.0135521, -0.0191355, 0.0148156, -0.0336579, 0.0326876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 245
type: A, layer: 3, pos: 190
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 249
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 103
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 129

Time for candidate selection: 4.07 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0532431, upper bound: 0.0657680
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0657268, upper bound: 0.0660450
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0168535, 0.0107802, -0.0169868, 0.0104848, -0.0273384, 0.0277671
1: -0.0259466, 0.0080101, -0.0252545, 0.0075937, -0.0335403, 0.0332646
2: 0.0309324, 0.0623009, 0.0313611, 0.0617922, -0.0308598, 0.0309398
3: -0.0011731, 0.0499757, -0.0014139, 0.0493159, -0.0433868, 0.0442417
4: -0.0197772, 0.0175995, -0.0191598, 0.0167609, -0.0365381, 0.0367593
5: -0.0025473, 0.0340948, -0.0018597, 0.0333340, -0.0358813, 0.0359545
6: -0.0448553, -0.0066582, -0.0441926, -0.0064532, -0.0384020, 0.0375344
7: 0.8686290, 0.9696742, 0.8697683, 0.9700685, -0.1014395, 0.0999058
8: -0.0105561, 0.0380835, -0.0108000, 0.0372911, -0.0478472, 0.0488835
9: -0.0191355, 0.0148156, -0.0188423, 0.0135521, -0.0326876, 0.0336579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0639154, upper bound: 0.0627026
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0639154, upper bound: 0.0669917
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0168535, 0.0107802, -0.0168535, 0.0107802, -0.0276338, 0.0276338
1: -0.0259466, 0.0080101, -0.0259466, 0.0080101, -0.0339567, 0.0339567
2: 0.0309324, 0.0623009, 0.0309324, 0.0623009, -0.0313685, 0.0313685
3: -0.0011731, 0.0499757, -0.0011731, 0.0499757, -0.0440909, 0.0440909
4: -0.0197772, 0.0175995, -0.0197772, 0.0175995, -0.0373767, 0.0373767
5: -0.0025473, 0.0340948, -0.0025473, 0.0340948, -0.0366421, 0.0366421
6: -0.0448553, -0.0066582, -0.0448553, -0.0066582, -0.0381971, 0.0381971
7: 0.8686290, 0.9696742, 0.8686290, 0.9696742, -0.1010452, 0.1010452
8: -0.0105561, 0.0380835, -0.0105561, 0.0380835, -0.0486396, 0.0486396
9: -0.0191355, 0.0148156, -0.0191355, 0.0148156, -0.0339511, 0.0339511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0507827, upper bound: 0.0650248
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0595080, upper bound: 0.0674736
time: 0.57 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.73 seconds
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0552086, upper bound: 0.0658428
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0661403, upper bound: 0.0661415
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0552086, upper bound: 0.0661259
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0661403, upper bound: 0.0665076
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0663063, upper bound: 0.0647610
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0663063, upper bound: 0.0673213
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0566410, upper bound: 0.0664480
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0625574, upper bound: 0.0677959
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0545920, upper bound: 0.0655103
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0659800, upper bound: 0.0657268
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0545920, upper bound: 0.0657510
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0659800, upper bound: 0.0660391
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0656925, upper bound: 0.0628824
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0656925, upper bound: 0.0669910
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0526818, upper bound: 0.0650821
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0601608, upper bound: 0.0674736
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0532431, upper bound: 0.0656988
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0657268, upper bound: 0.0659828
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0532431, upper bound: 0.0659990
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0657268, upper bound: 0.0663497
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0639169, upper bound: 0.0636421
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0639169, upper bound: 0.0672004
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0529687, upper bound: 0.0655753
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0602970, upper bound: 0.0676396
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0532431, upper bound: 0.0655135
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0657268, upper bound: 0.0657268
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0532431, upper bound: 0.0657680
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0657268, upper bound: 0.0660450
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0639154, upper bound: 0.0627026
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0639154, upper bound: 0.0669917
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0507827, upper bound: 0.0650248
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 7, lower bound: -0.0595080, upper bound: 0.0674736

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0166611, 0.0112066, -0.0167800, 0.0109073, -0.0275684, 0.0279866
1: -0.0269457, 0.0086112, -0.0262443, 0.0081892, -0.0351350, 0.0348555
2: 0.0303136, 0.0630351, 0.0307480, 0.0625196, -0.0322060, 0.0322870
3: -0.0008257, 0.0509283, -0.0010405, 0.0502595, -0.0438771, 0.0447479
4: -0.0206683, 0.0188100, -0.0200426, 0.0179601, -0.0386284, 0.0388526
5: -0.0035398, 0.0351930, -0.0028429, 0.0344220, -0.0379617, 0.0380360
6: -0.0458120, -0.0069539, -0.0451403, -0.0067712, -0.0390408, 0.0381864
7: 0.8669848, 0.9691050, 0.8681393, 0.9694566, -0.1024717, 0.1009657
8: -0.0102041, 0.0392275, -0.0104216, 0.0384243, -0.0486284, 0.0496491
9: -0.0195589, 0.0166392, -0.0192617, 0.0153589, -0.0349178, 0.0359009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 245
type: B, layer: 3, pos: 190
type: B, layer: 3, pos: 249
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 103
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 129

Time for candidate selection: 4.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0627732, upper bound: 0.0552086
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0630109, upper bound: 0.0661403
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0166611, 0.0112066, -0.0166697, 0.0112259, -0.0278870, 0.0278763
1: -0.0269457, 0.0086112, -0.0269909, 0.0086384, -0.0355842, 0.0356021
2: 0.0303136, 0.0630351, 0.0302856, 0.0630682, -0.0327547, 0.0327494
3: -0.0008257, 0.0509283, -0.0008411, 0.0509714, -0.0446218, 0.0445971
4: -0.0206683, 0.0188100, -0.0207086, 0.0188646, -0.0395329, 0.0395185
5: -0.0035398, 0.0351930, -0.0035845, 0.0352427, -0.0387824, 0.0387776
6: -0.0458120, -0.0069539, -0.0458552, -0.0069408, -0.0388711, 0.0389013
7: 0.8669848, 0.9691050, 0.8669106, 0.9691300, -0.1021452, 0.1021944
8: -0.0102041, 0.0392275, -0.0102197, 0.0392792, -0.0494833, 0.0494472
9: -0.0195589, 0.0166392, -0.0195780, 0.0167216, -0.0362805, 0.0362173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0562713, upper bound: 0.0572084
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0466804, upper bound: 0.0467415
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0166611, 0.0112066, -0.0169868, 0.0104848, -0.0271460, 0.0281934
1: -0.0269457, 0.0086112, -0.0252545, 0.0075937, -0.0345395, 0.0338657
2: 0.0303136, 0.0630351, 0.0313611, 0.0617922, -0.0314786, 0.0316740
3: -0.0008257, 0.0509283, -0.0014139, 0.0493159, -0.0430070, 0.0451422
4: -0.0206683, 0.0188100, -0.0191598, 0.0167609, -0.0374292, 0.0379698
5: -0.0035398, 0.0351930, -0.0018597, 0.0333340, -0.0368737, 0.0370528
6: -0.0458120, -0.0069539, -0.0441926, -0.0064532, -0.0393587, 0.0372386
7: 0.8669848, 0.9691050, 0.8697683, 0.9700685, -0.1030836, 0.0993367
8: -0.0102041, 0.0392275, -0.0108000, 0.0372911, -0.0474951, 0.0500275
9: -0.0195589, 0.0166392, -0.0188423, 0.0135521, -0.0331110, 0.0354815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 245
type: B, layer: 3, pos: 190
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 249
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 103
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 129

Time for candidate selection: 4.10 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0605658, upper bound: 0.0532431
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0613559, upper bound: 0.0657274
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0166611, 0.0112066, -0.0168535, 0.0107802, -0.0274414, 0.0280601
1: -0.0269457, 0.0086112, -0.0259466, 0.0080101, -0.0349559, 0.0345578
2: 0.0303136, 0.0630351, 0.0309324, 0.0623009, -0.0319873, 0.0321027
3: -0.0008257, 0.0509283, -0.0011731, 0.0499757, -0.0437235, 0.0449629
4: -0.0206683, 0.0188100, -0.0197772, 0.0175995, -0.0382678, 0.0385871
5: -0.0035398, 0.0351930, -0.0025473, 0.0340948, -0.0376345, 0.0377403
6: -0.0458120, -0.0069539, -0.0448553, -0.0066582, -0.0391538, 0.0379014
7: 0.8669848, 0.9691050, 0.8686290, 0.9696742, -0.1026893, 0.1004760
8: -0.0102041, 0.0392275, -0.0105561, 0.0380835, -0.0482876, 0.0497836
9: -0.0195589, 0.0166392, -0.0191355, 0.0148156, -0.0343745, 0.0357748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0558388, upper bound: 0.0556621
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0464745, upper bound: 0.0454009
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0168443, 0.0107622, -0.0167800, 0.0109073, -0.0277516, 0.0275422
1: -0.0259042, 0.0079846, -0.0262443, 0.0081892, -0.0340934, 0.0342289
2: 0.0309586, 0.0622697, 0.0307480, 0.0625196, -0.0315610, 0.0315217
3: -0.0011565, 0.0499353, -0.0010405, 0.0502595, -0.0442419, 0.0438444
4: -0.0197393, 0.0175481, -0.0200426, 0.0179601, -0.0376994, 0.0375907
5: -0.0025052, 0.0340481, -0.0028429, 0.0344220, -0.0369271, 0.0368911
6: -0.0448147, -0.0066723, -0.0451403, -0.0067712, -0.0380435, 0.0384680
7: 0.8686990, 0.9696468, 0.8681393, 0.9694566, -0.1007575, 0.1015075
8: -0.0105393, 0.0380350, -0.0104216, 0.0384243, -0.0489636, 0.0484566
9: -0.0191175, 0.0147381, -0.0192617, 0.0153589, -0.0344765, 0.0339998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 245
type: B, layer: 3, pos: 190
type: B, layer: 3, pos: 249
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 103
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 129

Time for candidate selection: 4.05 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0590328, upper bound: 0.0545920
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0597573, upper bound: 0.0659800
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0168443, 0.0107622, -0.0166697, 0.0112259, -0.0280702, 0.0274318
1: -0.0259042, 0.0079846, -0.0269909, 0.0086384, -0.0345426, 0.0349755
2: 0.0309586, 0.0622697, 0.0302856, 0.0630682, -0.0321096, 0.0319841
3: -0.0011565, 0.0499353, -0.0008411, 0.0509714, -0.0449888, 0.0436964
4: -0.0197393, 0.0175481, -0.0207086, 0.0188646, -0.0386040, 0.0382567
5: -0.0025052, 0.0340481, -0.0035845, 0.0352427, -0.0377478, 0.0376327
6: -0.0448147, -0.0066723, -0.0458552, -0.0069408, -0.0378738, 0.0391829
7: 0.8686990, 0.9696468, 0.8669106, 0.9691300, -0.1004310, 0.1027362
8: -0.0105393, 0.0380350, -0.0102197, 0.0392792, -0.0498184, 0.0482547
9: -0.0191175, 0.0147381, -0.0195780, 0.0167216, -0.0358392, 0.0343161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0551998, upper bound: 0.0571396
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0452378, upper bound: 0.0465322
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0168443, 0.0107622, -0.0169868, 0.0104848, -0.0273292, 0.0277490
1: -0.0259042, 0.0079846, -0.0252545, 0.0075937, -0.0334979, 0.0332391
2: 0.0309586, 0.0622697, 0.0313611, 0.0617922, -0.0308336, 0.0309086
3: -0.0011565, 0.0499353, -0.0014139, 0.0493159, -0.0433462, 0.0441968
4: -0.0197393, 0.0175481, -0.0191598, 0.0167609, -0.0365003, 0.0367079
5: -0.0025052, 0.0340481, -0.0018597, 0.0333340, -0.0358392, 0.0359079
6: -0.0448147, -0.0066723, -0.0441926, -0.0064532, -0.0383614, 0.0375203
7: 0.8686990, 0.9696468, 0.8697683, 0.9700685, -0.1013694, 0.0998785
8: -0.0105393, 0.0380350, -0.0108000, 0.0372911, -0.0478303, 0.0488350
9: -0.0191175, 0.0147381, -0.0188423, 0.0135521, -0.0326696, 0.0335804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 245
type: B, layer: 3, pos: 190
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 249
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 103
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 129

Time for candidate selection: 4.17 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0582546, upper bound: 0.0532431
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0593554, upper bound: 0.0657274
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0168443, 0.0107622, -0.0168535, 0.0107802, -0.0276246, 0.0276157
1: -0.0259042, 0.0079846, -0.0259466, 0.0080101, -0.0339143, 0.0339312
2: 0.0309586, 0.0622697, 0.0309324, 0.0623009, -0.0313422, 0.0313373
3: -0.0011565, 0.0499353, -0.0011731, 0.0499757, -0.0440466, 0.0440202
4: -0.0197393, 0.0175481, -0.0197772, 0.0175995, -0.0373389, 0.0373253
5: -0.0025052, 0.0340481, -0.0025473, 0.0340948, -0.0366000, 0.0365954
6: -0.0448147, -0.0066723, -0.0448553, -0.0066582, -0.0381565, 0.0381830
7: 0.8686990, 0.9696468, 0.8686290, 0.9696742, -0.1009752, 0.1010178
8: -0.0105393, 0.0380350, -0.0105561, 0.0380835, -0.0486228, 0.0485911
9: -0.0191175, 0.0147381, -0.0191355, 0.0148156, -0.0339331, 0.0338736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0551028, upper bound: 0.0556890
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0452378, upper bound: 0.0454030
time: 0.53 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.75 seconds
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.75
Output dim: 7, lower bound: -0.0627732, upper bound: 0.0552086
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.75
Output dim: 7, lower bound: -0.0630109, upper bound: 0.0661403
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.75
Output dim: 7, lower bound: -0.0562713, upper bound: 0.0572084
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.75
Output dim: 7, lower bound: -0.0466804, upper bound: 0.0467415
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.75
Output dim: 7, lower bound: -0.0605658, upper bound: 0.0532431
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.75
Output dim: 7, lower bound: -0.0613559, upper bound: 0.0657274
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.75
Output dim: 7, lower bound: -0.0558388, upper bound: 0.0556621
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.75
Output dim: 7, lower bound: -0.0464745, upper bound: 0.0454009
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.75
Output dim: 7, lower bound: -0.0590328, upper bound: 0.0545920
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.75
Output dim: 7, lower bound: -0.0597573, upper bound: 0.0659800
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.75
Output dim: 7, lower bound: -0.0551998, upper bound: 0.0571396
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.75
Output dim: 7, lower bound: -0.0452378, upper bound: 0.0465322
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.75
Output dim: 7, lower bound: -0.0582546, upper bound: 0.0532431
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.75
Output dim: 7, lower bound: -0.0593554, upper bound: 0.0657274
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.75
Output dim: 7, lower bound: -0.0551028, upper bound: 0.0556890
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.75
Output dim: 7, lower bound: -0.0452378, upper bound: 0.0454030

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.01 + 271.90 = 274.91 seconds
