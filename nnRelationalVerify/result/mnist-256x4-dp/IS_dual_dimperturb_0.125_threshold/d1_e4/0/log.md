## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 4.776e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0034553, -0.0028078, -0.0034553, -0.0028078, -0.0002429, 0.0002429)
1: (-0.0045188, -0.0043970, -0.0045188, -0.0043970, -0.0000420, 0.0000420)
2: (0.0101708, 0.0109970, 0.0101708, 0.0109970, -0.0003061, 0.0003061)
3: (1.0087261, 1.0089240, 1.0087261, 1.0089240, -0.0000747, 0.0000747)
4: (-0.0034118, -0.0032837, -0.0034118, -0.0032837, -0.0000468, 0.0000468)
5: (0.0013063, 0.0018017, 0.0013063, 0.0018017, -0.0001855, 0.0001855)
6: (-0.0025218, -0.0024970, -0.0025218, -0.0024970, -0.0000106, 0.0000106)
7: (-0.0088339, -0.0077199, -0.0088339, -0.0077199, -0.0004434, 0.0004434)
8: (-0.0044941, -0.0031546, -0.0044941, -0.0031546, -0.0004846, 0.0004846)
9: (-0.0026307, -0.0019956, -0.0026307, -0.0019956, -0.0002270, 0.0002270)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.44 + 1.27 = 2.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0000585, upper bound: 0.0000585

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000535, upper bound: 0.0000554
time: 0.46 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000554
time: 0.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.08 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.08
Output dim: 3, lower bound: -0.0000535, upper bound: 0.0000554
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.08
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000554

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0034551, -0.0028305, -0.0034553, -0.0028157, -0.0002307, 0.0002118
1: -0.0045187, -0.0044024, -0.0045187, -0.0043988, -0.0000388, 0.0000337
2: 0.0101710, 0.0109665, 0.0101709, 0.0109866, -0.0002896, 0.0002640
3: 1.0087337, 1.0089238, 1.0087287, 1.0089239, -0.0000631, 0.0000704
4: -0.0034069, -0.0032837, -0.0034101, -0.0032837, -0.0000398, 0.0000441
5: 0.0013064, 0.0017842, 0.0013063, 0.0017956, -0.0001761, 0.0001615
6: -0.0025218, -0.0024971, -0.0025218, -0.0024971, -0.0000105, 0.0000104
7: -0.0088041, -0.0077202, -0.0088236, -0.0077200, -0.0004024, 0.0004275
8: -0.0044416, -0.0031550, -0.0044761, -0.0031547, -0.0004088, 0.0004554
9: -0.0026305, -0.0020217, -0.0026306, -0.0020047, -0.0002125, 0.0001893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000527
time: 0.46 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000527
time: 0.46 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0034675, -0.0028295, -0.0034552, -0.0028164, -0.0002749, 0.0002166
1: -0.0045225, -0.0044014, -0.0045187, -0.0043987, -0.0000516, 0.0000349
2: 0.0101542, 0.0109685, 0.0101709, 0.0109857, -0.0003499, 0.0002702
3: 1.0087334, 1.0089282, 1.0087290, 1.0089239, -0.0000643, 0.0000871
4: -0.0034073, -0.0032809, -0.0034100, -0.0032837, -0.0000408, 0.0000541
5: 0.0012968, 0.0017850, 0.0013063, 0.0017951, -0.0002102, 0.0001652
6: -0.0025218, -0.0024971, -0.0025218, -0.0024971, -0.0000106, 0.0000105
7: -0.0088009, -0.0077110, -0.0088203, -0.0077201, -0.0004074, 0.0004803
8: -0.0044476, -0.0031237, -0.0044753, -0.0031548, -0.0004201, 0.0005659
9: -0.0026474, -0.0020173, -0.0026306, -0.0020041, -0.0002680, 0.0001949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000527
time: 0.47 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000527, upper bound: 0.0000527
time: 0.45 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.45 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000527
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000527
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000527
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 3, lower bound: -0.0000527, upper bound: 0.0000527

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0034551, -0.0028305, -0.0034550, -0.0028238, -0.0002188, 0.0002117
1: -0.0045187, -0.0044024, -0.0045187, -0.0043989, -0.0000387, 0.0000337
2: 0.0101710, 0.0109665, 0.0101712, 0.0109774, -0.0002762, 0.0002639
3: 1.0087337, 1.0089238, 1.0087287, 1.0089203, -0.0000583, 0.0000704
4: -0.0034069, -0.0032837, -0.0034089, -0.0032838, -0.0000398, 0.0000423
5: 0.0013064, 0.0017842, 0.0013065, 0.0017895, -0.0001671, 0.0001614
6: -0.0025218, -0.0024971, -0.0025218, -0.0024979, -0.0000094, 0.0000104
7: -0.0088041, -0.0077202, -0.0088023, -0.0077204, -0.0004023, 0.0003969
8: -0.0044416, -0.0031550, -0.0044649, -0.0031553, -0.0004086, 0.0004390
9: -0.0026305, -0.0020217, -0.0026304, -0.0020088, -0.0002061, 0.0001892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000515
time: 0.45 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000527
time: 0.45 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0034550, -0.0028410, -0.0034839, -0.0028447, -0.0002221, 0.0002617
1: -0.0045186, -0.0044025, -0.0045193, -0.0043991, -0.0000387, 0.0000346
2: 0.0101713, 0.0109546, 0.0101385, 0.0109532, -0.0002794, 0.0003201
3: 1.0087337, 1.0089196, 1.0087162, 1.0089140, -0.0000618, 0.0000917
4: -0.0034052, -0.0032838, -0.0034057, -0.0032795, -0.0000471, 0.0000426
5: 0.0013065, 0.0017763, 0.0012847, 0.0017737, -0.0001696, 0.0001991
6: -0.0025218, -0.0024981, -0.0025250, -0.0024993, -0.0000101, 0.0000155
7: -0.0087771, -0.0077204, -0.0087563, -0.0076433, -0.0005338, 0.0004100
8: -0.0044264, -0.0031554, -0.0044353, -0.0031160, -0.0004754, 0.0004412
9: -0.0026303, -0.0020274, -0.0026466, -0.0020205, -0.0002067, 0.0002151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000515
time: 0.44 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000527
time: 0.44 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0034675, -0.0028295, -0.0034550, -0.0028245, -0.0002631, 0.0002165
1: -0.0045225, -0.0044014, -0.0045187, -0.0043988, -0.0000515, 0.0000348
2: 0.0101542, 0.0109685, 0.0101713, 0.0109765, -0.0003366, 0.0002700
3: 1.0087334, 1.0089282, 1.0087290, 1.0089203, -0.0000595, 0.0000871
4: -0.0034073, -0.0032809, -0.0034088, -0.0032838, -0.0000408, 0.0000524
5: 0.0012968, 0.0017850, 0.0013065, 0.0017890, -0.0002013, 0.0001651
6: -0.0025218, -0.0024971, -0.0025218, -0.0024979, -0.0000095, 0.0000105
7: -0.0088009, -0.0077110, -0.0087986, -0.0077204, -0.0004072, 0.0004505
8: -0.0044476, -0.0031237, -0.0044649, -0.0031554, -0.0004198, 0.0005498
9: -0.0026474, -0.0020173, -0.0026303, -0.0020084, -0.0002618, 0.0001947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000511
time: 0.45 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000511
time: 0.46 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0034673, -0.0028404, -0.0034839, -0.0028462, -0.0002658, 0.0002664
1: -0.0045224, -0.0044015, -0.0045193, -0.0043990, -0.0000515, 0.0000357
2: 0.0101544, 0.0109559, 0.0101386, 0.0109517, -0.0003392, 0.0003261
3: 1.0087334, 1.0089236, 1.0087168, 1.0089141, -0.0000628, 0.0001025
4: -0.0034056, -0.0032810, -0.0034055, -0.0032795, -0.0000481, 0.0000527
5: 0.0012970, 0.0017768, 0.0012848, 0.0017725, -0.0002034, 0.0002027
6: -0.0025218, -0.0024981, -0.0025250, -0.0024993, -0.0000101, 0.0000156
7: -0.0087758, -0.0077113, -0.0087547, -0.0076434, -0.0005385, 0.0004629
8: -0.0044314, -0.0031242, -0.0044336, -0.0031160, -0.0004865, 0.0005518
9: -0.0026472, -0.0020234, -0.0026466, -0.0020208, -0.0002623, 0.0002206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000527, upper bound: 0.0000511
time: 0.49 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000527, upper bound: 0.0000511
time: 0.47 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.62 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000515
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000527
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000515
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000527
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000511
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000511
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 3, lower bound: -0.0000527, upper bound: 0.0000511
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 3, lower bound: -0.0000527, upper bound: 0.0000511

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0034551, -0.0028305, -0.0034549, -0.0028388, -0.0002005, 0.0002116
1: -0.0045187, -0.0044024, -0.0045186, -0.0044025, -0.0000336, 0.0000337
2: 0.0101710, 0.0109665, 0.0101714, 0.0109573, -0.0002513, 0.0002638
3: 1.0087337, 1.0089238, 1.0087337, 1.0089202, -0.0000582, 0.0000630
4: -0.0034069, -0.0032837, -0.0034057, -0.0032838, -0.0000398, 0.0000382
5: 0.0013064, 0.0017842, 0.0013066, 0.0017779, -0.0001530, 0.0001614
6: -0.0025218, -0.0024971, -0.0025218, -0.0024980, -0.0000093, 0.0000104
7: -0.0088041, -0.0077202, -0.0087835, -0.0077205, -0.0004022, 0.0003732
8: -0.0044416, -0.0031550, -0.0044305, -0.0031555, -0.0004084, 0.0003937
9: -0.0026305, -0.0020217, -0.0026302, -0.0020260, -0.0001835, 0.0001891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000506
time: 0.46 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000515
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0034551, -0.0028305, -0.0034673, -0.0028374, -0.0002277, 0.0002451
1: -0.0045187, -0.0044024, -0.0045224, -0.0044014, -0.0000418, 0.0000434
2: 0.0101710, 0.0109665, 0.0101545, 0.0109598, -0.0002890, 0.0003093
3: 1.0087337, 1.0089238, 1.0087334, 1.0089248, -0.0000719, 0.0000743
4: -0.0034069, -0.0032837, -0.0034062, -0.0032810, -0.0000474, 0.0000445
5: 0.0013064, 0.0017842, 0.0012970, 0.0017790, -0.0001740, 0.0001872
6: -0.0025218, -0.0024971, -0.0025218, -0.0024980, -0.0000094, 0.0000105
7: -0.0088041, -0.0077202, -0.0087794, -0.0077113, -0.0004441, 0.0004026
8: -0.0044416, -0.0031550, -0.0044373, -0.0031243, -0.0004917, 0.0004645
9: -0.0026305, -0.0020217, -0.0026472, -0.0020215, -0.0002194, 0.0002309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000519
time: 0.47 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000527
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0034550, -0.0028410, -0.0034838, -0.0028593, -0.0002044, 0.0002616
1: -0.0045186, -0.0044025, -0.0045193, -0.0044027, -0.0000336, 0.0000346
2: 0.0101713, 0.0109546, 0.0101387, 0.0109340, -0.0002553, 0.0003200
3: 1.0087337, 1.0089196, 1.0087209, 1.0089138, -0.0000618, 0.0000868
4: -0.0034052, -0.0032838, -0.0034025, -0.0032795, -0.0000471, 0.0000386
5: 0.0013065, 0.0017763, 0.0012848, 0.0017625, -0.0001559, 0.0001990
6: -0.0025218, -0.0024981, -0.0025250, -0.0024994, -0.0000100, 0.0000155
7: -0.0087771, -0.0077204, -0.0087354, -0.0076435, -0.0005337, 0.0003875
8: -0.0044264, -0.0031554, -0.0044008, -0.0031162, -0.0004753, 0.0003966
9: -0.0026303, -0.0020274, -0.0026465, -0.0020377, -0.0001843, 0.0002151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000504
time: 0.46 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000515
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0034550, -0.0028410, -0.0034943, -0.0028592, -0.0002304, 0.0002904
1: -0.0045186, -0.0044025, -0.0045231, -0.0044016, -0.0000418, 0.0000444
2: 0.0101713, 0.0109546, 0.0101243, 0.0109344, -0.0002917, 0.0003601
3: 1.0087337, 1.0089196, 1.0087214, 1.0089179, -0.0000758, 0.0000896
4: -0.0034052, -0.0032838, -0.0034027, -0.0032770, -0.0000540, 0.0000448
5: 0.0013065, 0.0017763, 0.0012767, 0.0017626, -0.0001761, 0.0002213
6: -0.0025218, -0.0024981, -0.0025250, -0.0024993, -0.0000101, 0.0000155
7: -0.0087771, -0.0077204, -0.0087369, -0.0076291, -0.0005665, 0.0004144
8: -0.0044264, -0.0031554, -0.0044047, -0.0030862, -0.0005515, 0.0004663
9: -0.0026303, -0.0020274, -0.0026624, -0.0020340, -0.0002199, 0.0002539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000519
time: 0.45 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000527
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0034675, -0.0028295, -0.0034549, -0.0028388, -0.0002340, 0.0002391
1: -0.0045225, -0.0044014, -0.0045186, -0.0044025, -0.0000433, 0.0000419
2: 0.0101542, 0.0109685, 0.0101714, 0.0109573, -0.0002968, 0.0003018
3: 1.0087334, 1.0089282, 1.0087337, 1.0089202, -0.0000695, 0.0000756
4: -0.0034073, -0.0032809, -0.0034057, -0.0032838, -0.0000462, 0.0000458
5: 0.0012968, 0.0017850, 0.0013066, 0.0017779, -0.0001788, 0.0001826
6: -0.0025218, -0.0024971, -0.0025218, -0.0024980, -0.0000094, 0.0000105
7: -0.0088009, -0.0077110, -0.0087835, -0.0077205, -0.0004327, 0.0004151
8: -0.0044476, -0.0031237, -0.0044305, -0.0031555, -0.0004796, 0.0004770
9: -0.0026474, -0.0020173, -0.0026302, -0.0020260, -0.0002253, 0.0002251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000500
time: 0.47 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000511
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0034675, -0.0028295, -0.0034673, -0.0028374, -0.0002057, 0.0002165
1: -0.0045225, -0.0044014, -0.0045224, -0.0044014, -0.0000349, 0.0000349
2: 0.0101542, 0.0109685, 0.0101545, 0.0109598, -0.0002578, 0.0002701
3: 1.0087334, 1.0089282, 1.0087334, 1.0089248, -0.0000603, 0.0000649
4: -0.0034073, -0.0032809, -0.0034062, -0.0032810, -0.0000408, 0.0000392
5: 0.0012968, 0.0017850, 0.0012970, 0.0017790, -0.0001570, 0.0001651
6: -0.0025218, -0.0024971, -0.0025218, -0.0024980, -0.0000094, 0.0000105
7: -0.0088009, -0.0077110, -0.0087794, -0.0077113, -0.0004072, 0.0003790
8: -0.0044476, -0.0031237, -0.0044373, -0.0031243, -0.0004199, 0.0004053
9: -0.0026474, -0.0020173, -0.0026472, -0.0020215, -0.0001892, 0.0001948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000500
time: 0.49 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000511
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0034673, -0.0028404, -0.0034838, -0.0028593, -0.0002379, 0.0002864
1: -0.0045224, -0.0044015, -0.0045193, -0.0044027, -0.0000433, 0.0000427
2: 0.0101544, 0.0109559, 0.0101387, 0.0109340, -0.0003008, 0.0003548
3: 1.0087334, 1.0089236, 1.0087209, 1.0089138, -0.0000731, 0.0000970
4: -0.0034056, -0.0032810, -0.0034025, -0.0032795, -0.0000531, 0.0000462
5: 0.0012970, 0.0017768, 0.0012848, 0.0017625, -0.0001818, 0.0002182
6: -0.0025218, -0.0024981, -0.0025250, -0.0024994, -0.0000101, 0.0000155
7: -0.0087758, -0.0077113, -0.0087354, -0.0076435, -0.0005581, 0.0004295
8: -0.0044314, -0.0031242, -0.0044008, -0.0031162, -0.0005423, 0.0004799
9: -0.0026472, -0.0020234, -0.0026465, -0.0020377, -0.0002260, 0.0002496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000497
time: 0.46 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000497
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0034673, -0.0028404, -0.0034943, -0.0028592, -0.0002097, 0.0002665
1: -0.0045224, -0.0044015, -0.0045231, -0.0044016, -0.0000348, 0.0000358
2: 0.0101544, 0.0109559, 0.0101243, 0.0109344, -0.0002617, 0.0003262
3: 1.0087334, 1.0089236, 1.0087214, 1.0089179, -0.0000640, 0.0000887
4: -0.0034056, -0.0032810, -0.0034027, -0.0032770, -0.0000482, 0.0000395
5: 0.0012970, 0.0017768, 0.0012767, 0.0017626, -0.0001599, 0.0002027
6: -0.0025218, -0.0024981, -0.0025250, -0.0024993, -0.0000102, 0.0000156
7: -0.0087758, -0.0077113, -0.0087369, -0.0076291, -0.0005387, 0.0003960
8: -0.0044314, -0.0031242, -0.0044047, -0.0030862, -0.0004867, 0.0004071
9: -0.0026472, -0.0020234, -0.0026624, -0.0020340, -0.0001899, 0.0002207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000497
time: 0.46 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000511
time: 0.47 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.49 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000506
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000515
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000519
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000527
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000504
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000515
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000519
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000497, upper bound: 0.0000527
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000500
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000511
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000500
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000511
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000497
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000497
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000497
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000511

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0034549, -0.0028388, -0.0034549, -0.0028388, -0.0002003, 0.0002003
1: -0.0045186, -0.0044025, -0.0045186, -0.0044025, -0.0000336, 0.0000336
2: 0.0101714, 0.0109573, 0.0101714, 0.0109573, -0.0002511, 0.0002511
3: 1.0087337, 1.0089202, 1.0087337, 1.0089202, -0.0000582, 0.0000582
4: -0.0034057, -0.0032838, -0.0034057, -0.0032838, -0.0000381, 0.0000381
5: 0.0013066, 0.0017779, 0.0013066, 0.0017779, -0.0001529, 0.0001529
6: -0.0025218, -0.0024980, -0.0025218, -0.0024980, -0.0000093, 0.0000093
7: -0.0087835, -0.0077205, -0.0087835, -0.0077205, -0.0003730, 0.0003730
8: -0.0044305, -0.0031555, -0.0044305, -0.0031555, -0.0003935, 0.0003935
9: -0.0026302, -0.0020260, -0.0026302, -0.0020260, -0.0001833, 0.0001833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000485
time: 0.46 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000503
time: 0.47 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0034838, -0.0028593, -0.0034549, -0.0028388, -0.0002538, 0.0002060
1: -0.0045193, -0.0044027, -0.0045186, -0.0044025, -0.0000346, 0.0000335
2: 0.0101387, 0.0109340, 0.0101714, 0.0109573, -0.0003113, 0.0002572
3: 1.0087209, 1.0089138, 1.0087337, 1.0089202, -0.0000835, 0.0000605
4: -0.0034025, -0.0032795, -0.0034057, -0.0032838, -0.0000388, 0.0000460
5: 0.0012848, 0.0017625, 0.0013066, 0.0017779, -0.0001931, 0.0001571
6: -0.0025250, -0.0024994, -0.0025218, -0.0024980, -0.0000146, 0.0000100
7: -0.0087354, -0.0076435, -0.0087835, -0.0077205, -0.0003893, 0.0005124
8: -0.0044008, -0.0031162, -0.0044305, -0.0031555, -0.0003992, 0.0004657
9: -0.0026465, -0.0020377, -0.0026302, -0.0020260, -0.0002116, 0.0001851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000483, upper bound: 0.0000512
time: 0.47 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000512
time: 0.46 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0034549, -0.0028388, -0.0034673, -0.0028374, -0.0002275, 0.0002338
1: -0.0045186, -0.0044025, -0.0045224, -0.0044014, -0.0000418, 0.0000433
2: 0.0101714, 0.0109573, 0.0101545, 0.0109598, -0.0002888, 0.0002966
3: 1.0087337, 1.0089202, 1.0087334, 1.0089248, -0.0000719, 0.0000695
4: -0.0034057, -0.0032838, -0.0034062, -0.0032810, -0.0000457, 0.0000445
5: 0.0013066, 0.0017779, 0.0012970, 0.0017790, -0.0001740, 0.0001787
6: -0.0025218, -0.0024980, -0.0025218, -0.0024980, -0.0000094, 0.0000094
7: -0.0087835, -0.0077205, -0.0087794, -0.0077113, -0.0004149, 0.0004024
8: -0.0044305, -0.0031555, -0.0044373, -0.0031243, -0.0004767, 0.0004642
9: -0.0026302, -0.0020260, -0.0026472, -0.0020215, -0.0002192, 0.0002251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000495, upper bound: 0.0000501
time: 0.46 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000495, upper bound: 0.0000516
time: 0.48 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0034838, -0.0028593, -0.0034673, -0.0028374, -0.0002810, 0.0002394
1: -0.0045193, -0.0044027, -0.0045224, -0.0044014, -0.0000427, 0.0000431
2: 0.0101387, 0.0109340, 0.0101545, 0.0109598, -0.0003490, 0.0003026
3: 1.0087209, 1.0089138, 1.0087334, 1.0089248, -0.0000972, 0.0000718
4: -0.0034025, -0.0032795, -0.0034062, -0.0032810, -0.0000464, 0.0000524
5: 0.0012848, 0.0017625, 0.0012970, 0.0017790, -0.0002142, 0.0001829
6: -0.0025250, -0.0024994, -0.0025218, -0.0024980, -0.0000147, 0.0000100
7: -0.0087354, -0.0076435, -0.0087794, -0.0077113, -0.0004313, 0.0005418
8: -0.0044008, -0.0031162, -0.0044373, -0.0031243, -0.0004825, 0.0005364
9: -0.0026465, -0.0020377, -0.0026472, -0.0020215, -0.0002475, 0.0002269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000479, upper bound: 0.0000524
time: 0.48 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000495, upper bound: 0.0000524
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0034549, -0.0028388, -0.0034838, -0.0028593, -0.0002060, 0.0002538
1: -0.0045186, -0.0044025, -0.0045193, -0.0044027, -0.0000335, 0.0000346
2: 0.0101714, 0.0109573, 0.0101387, 0.0109340, -0.0002572, 0.0003113
3: 1.0087337, 1.0089202, 1.0087209, 1.0089138, -0.0000605, 0.0000835
4: -0.0034057, -0.0032838, -0.0034025, -0.0032795, -0.0000460, 0.0000388
5: 0.0013066, 0.0017779, 0.0012848, 0.0017625, -0.0001571, 0.0001931
6: -0.0025218, -0.0024980, -0.0025250, -0.0024994, -0.0000100, 0.0000146
7: -0.0087835, -0.0077205, -0.0087354, -0.0076435, -0.0005124, 0.0003893
8: -0.0044305, -0.0031555, -0.0044008, -0.0031162, -0.0004657, 0.0003992
9: -0.0026302, -0.0020260, -0.0026465, -0.0020377, -0.0001851, 0.0002116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000483
time: 0.48 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000501
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0034838, -0.0028593, -0.0034838, -0.0028593, -0.0002064, 0.0002064
1: -0.0045193, -0.0044027, -0.0045193, -0.0044027, -0.0000343, 0.0000343
2: 0.0101387, 0.0109340, 0.0101387, 0.0109340, -0.0002578, 0.0002578
3: 1.0087209, 1.0089138, 1.0087209, 1.0089138, -0.0000618, 0.0000618
4: -0.0034025, -0.0032795, -0.0034025, -0.0032795, -0.0000390, 0.0000390
5: 0.0012848, 0.0017625, 0.0012848, 0.0017625, -0.0001574, 0.0001574
6: -0.0025250, -0.0024994, -0.0025250, -0.0024994, -0.0000100, 0.0000100
7: -0.0087354, -0.0076435, -0.0087354, -0.0076435, -0.0003903, 0.0003903
8: -0.0044008, -0.0031162, -0.0044008, -0.0031162, -0.0004011, 0.0004011
9: -0.0026465, -0.0020377, -0.0026465, -0.0020377, -0.0001866, 0.0001866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000499
time: 0.48 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000511
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0034549, -0.0028388, -0.0034943, -0.0028592, -0.0002250, 0.0002826
1: -0.0045186, -0.0044025, -0.0045231, -0.0044016, -0.0000417, 0.0000444
2: 0.0101714, 0.0109573, 0.0101243, 0.0109344, -0.0002851, 0.0003514
3: 1.0087337, 1.0089202, 1.0087214, 1.0089179, -0.0000688, 0.0000863
4: -0.0034057, -0.0032838, -0.0034027, -0.0032770, -0.0000529, 0.0000439
5: 0.0013066, 0.0017779, 0.0012767, 0.0017626, -0.0001720, 0.0002155
6: -0.0025218, -0.0024980, -0.0025250, -0.0024993, -0.0000101, 0.0000147
7: -0.0087835, -0.0077205, -0.0087369, -0.0076291, -0.0005452, 0.0004057
8: -0.0044305, -0.0031555, -0.0044047, -0.0030862, -0.0005419, 0.0004565
9: -0.0026302, -0.0020260, -0.0026624, -0.0020340, -0.0002153, 0.0002504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000495, upper bound: 0.0000501
time: 0.47 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000495, upper bound: 0.0000516
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0034838, -0.0028593, -0.0034943, -0.0028592, -0.0002323, 0.0002403
1: -0.0045193, -0.0044027, -0.0045231, -0.0044016, -0.0000425, 0.0000441
2: 0.0101387, 0.0109340, 0.0101243, 0.0109344, -0.0002942, 0.0003039
3: 1.0087209, 1.0089138, 1.0087214, 1.0089179, -0.0000758, 0.0000731
4: -0.0034025, -0.0032795, -0.0034027, -0.0032770, -0.0000467, 0.0000452
5: 0.0012848, 0.0017625, 0.0012767, 0.0017626, -0.0001775, 0.0001836
6: -0.0025250, -0.0024994, -0.0025250, -0.0024993, -0.0000101, 0.0000101
7: -0.0087354, -0.0076435, -0.0087369, -0.0076291, -0.0004331, 0.0004172
8: -0.0044008, -0.0031162, -0.0044047, -0.0030862, -0.0004854, 0.0004709
9: -0.0026465, -0.0020377, -0.0026624, -0.0020340, -0.0002222, 0.0002287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000495, upper bound: 0.0000501
time: 0.47 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000495, upper bound: 0.0000524
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0034673, -0.0028374, -0.0034549, -0.0028388, -0.0002338, 0.0002275
1: -0.0045224, -0.0044014, -0.0045186, -0.0044025, -0.0000433, 0.0000418
2: 0.0101545, 0.0109598, 0.0101714, 0.0109573, -0.0002966, 0.0002888
3: 1.0087334, 1.0089248, 1.0087337, 1.0089202, -0.0000695, 0.0000719
4: -0.0034062, -0.0032810, -0.0034057, -0.0032838, -0.0000445, 0.0000457
5: 0.0012970, 0.0017790, 0.0013066, 0.0017779, -0.0001787, 0.0001740
6: -0.0025218, -0.0024980, -0.0025218, -0.0024980, -0.0000094, 0.0000094
7: -0.0087794, -0.0077113, -0.0087835, -0.0077205, -0.0004024, 0.0004149
8: -0.0044373, -0.0031243, -0.0044305, -0.0031555, -0.0004642, 0.0004767
9: -0.0026472, -0.0020215, -0.0026302, -0.0020260, -0.0002251, 0.0002192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000497
time: 0.47 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000497
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0034943, -0.0028592, -0.0034549, -0.0028388, -0.0002826, 0.0002250
1: -0.0045231, -0.0044016, -0.0045186, -0.0044025, -0.0000444, 0.0000417
2: 0.0101243, 0.0109344, 0.0101714, 0.0109573, -0.0003514, 0.0002851
3: 1.0087214, 1.0089179, 1.0087337, 1.0089202, -0.0000863, 0.0000688
4: -0.0034027, -0.0032770, -0.0034057, -0.0032838, -0.0000439, 0.0000529
5: 0.0012767, 0.0017626, 0.0013066, 0.0017779, -0.0002155, 0.0001720
6: -0.0025250, -0.0024993, -0.0025218, -0.0024980, -0.0000147, 0.0000101
7: -0.0087369, -0.0076291, -0.0087835, -0.0077205, -0.0004057, 0.0005452
8: -0.0044047, -0.0030862, -0.0044305, -0.0031555, -0.0004565, 0.0005419
9: -0.0026624, -0.0020340, -0.0026302, -0.0020260, -0.0002504, 0.0002153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000507
time: 0.47 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000507
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0034673, -0.0028374, -0.0034673, -0.0028374, -0.0002056, 0.0002056
1: -0.0045224, -0.0044014, -0.0045224, -0.0044014, -0.0000348, 0.0000348
2: 0.0101545, 0.0109598, 0.0101545, 0.0109598, -0.0002577, 0.0002577
3: 1.0087334, 1.0089248, 1.0087334, 1.0089248, -0.0000603, 0.0000603
4: -0.0034062, -0.0032810, -0.0034062, -0.0032810, -0.0000392, 0.0000392
5: 0.0012970, 0.0017790, 0.0012970, 0.0017790, -0.0001569, 0.0001569
6: -0.0025218, -0.0024980, -0.0025218, -0.0024980, -0.0000094, 0.0000094
7: -0.0087794, -0.0077113, -0.0087794, -0.0077113, -0.0003788, 0.0003788
8: -0.0044373, -0.0031243, -0.0044373, -0.0031243, -0.0004051, 0.0004051
9: -0.0026472, -0.0020215, -0.0026472, -0.0020215, -0.0001891, 0.0001891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000481
time: 0.49 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000497
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0034943, -0.0028592, -0.0034673, -0.0028374, -0.0002591, 0.0002110
1: -0.0045231, -0.0044016, -0.0045224, -0.0044014, -0.0000358, 0.0000347
2: 0.0101243, 0.0109344, 0.0101545, 0.0109598, -0.0003180, 0.0002636
3: 1.0087214, 1.0089179, 1.0087334, 1.0089248, -0.0000856, 0.0000625
4: -0.0034027, -0.0032770, -0.0034062, -0.0032810, -0.0000399, 0.0000471
5: 0.0012767, 0.0017626, 0.0012970, 0.0017790, -0.0001972, 0.0001610
6: -0.0025250, -0.0024993, -0.0025218, -0.0024980, -0.0000147, 0.0000101
7: -0.0087369, -0.0076291, -0.0087794, -0.0077113, -0.0003950, 0.0005182
8: -0.0044047, -0.0030862, -0.0044373, -0.0031243, -0.0004108, 0.0004775
9: -0.0026624, -0.0020340, -0.0026472, -0.0020215, -0.0002174, 0.0001908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000507
time: 0.48 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000507
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0034673, -0.0028374, -0.0034838, -0.0028593, -0.0002394, 0.0002810
1: -0.0045224, -0.0044014, -0.0045193, -0.0044027, -0.0000431, 0.0000427
2: 0.0101545, 0.0109598, 0.0101387, 0.0109340, -0.0003026, 0.0003490
3: 1.0087334, 1.0089248, 1.0087209, 1.0089138, -0.0000718, 0.0000972
4: -0.0034062, -0.0032810, -0.0034025, -0.0032795, -0.0000524, 0.0000464
5: 0.0012970, 0.0017790, 0.0012848, 0.0017625, -0.0001829, 0.0002142
6: -0.0025218, -0.0024980, -0.0025250, -0.0024994, -0.0000100, 0.0000147
7: -0.0087794, -0.0077113, -0.0087354, -0.0076435, -0.0005418, 0.0004313
8: -0.0044373, -0.0031243, -0.0044008, -0.0031162, -0.0005364, 0.0004825
9: -0.0026472, -0.0020215, -0.0026465, -0.0020377, -0.0002269, 0.0002475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000479
time: 0.46 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000495
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0034943, -0.0028592, -0.0034838, -0.0028593, -0.0002403, 0.0002323
1: -0.0045231, -0.0044016, -0.0045193, -0.0044027, -0.0000441, 0.0000425
2: 0.0101243, 0.0109344, 0.0101387, 0.0109340, -0.0003039, 0.0002942
3: 1.0087214, 1.0089179, 1.0087209, 1.0089138, -0.0000731, 0.0000758
4: -0.0034027, -0.0032770, -0.0034025, -0.0032795, -0.0000452, 0.0000467
5: 0.0012767, 0.0017626, 0.0012848, 0.0017625, -0.0001836, 0.0001775
6: -0.0025250, -0.0024993, -0.0025250, -0.0024994, -0.0000101, 0.0000101
7: -0.0087369, -0.0076291, -0.0087354, -0.0076435, -0.0004172, 0.0004331
8: -0.0044047, -0.0030862, -0.0044008, -0.0031162, -0.0004709, 0.0004854
9: -0.0026624, -0.0020340, -0.0026465, -0.0020377, -0.0002287, 0.0002222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000507
time: 0.47 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000507
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0034673, -0.0028374, -0.0034943, -0.0028592, -0.0002110, 0.0002591
1: -0.0045224, -0.0044014, -0.0045231, -0.0044016, -0.0000347, 0.0000358
2: 0.0101545, 0.0109598, 0.0101243, 0.0109344, -0.0002636, 0.0003180
3: 1.0087334, 1.0089248, 1.0087214, 1.0089179, -0.0000625, 0.0000856
4: -0.0034062, -0.0032810, -0.0034027, -0.0032770, -0.0000471, 0.0000399
5: 0.0012970, 0.0017790, 0.0012767, 0.0017626, -0.0001610, 0.0001972
6: -0.0025218, -0.0024980, -0.0025250, -0.0024993, -0.0000101, 0.0000147
7: -0.0087794, -0.0077113, -0.0087369, -0.0076291, -0.0005182, 0.0003950
8: -0.0044373, -0.0031243, -0.0044047, -0.0030862, -0.0004775, 0.0004108
9: -0.0026472, -0.0020215, -0.0026624, -0.0020340, -0.0001908, 0.0002174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000479
time: 0.50 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000495
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0034943, -0.0028592, -0.0034943, -0.0028592, -0.0002116, 0.0002116
1: -0.0045231, -0.0044016, -0.0045231, -0.0044016, -0.0000356, 0.0000356
2: 0.0101243, 0.0109344, 0.0101243, 0.0109344, -0.0002643, 0.0002643
3: 1.0087214, 1.0089179, 1.0087214, 1.0089179, -0.0000640, 0.0000640
4: -0.0034027, -0.0032770, -0.0034027, -0.0032770, -0.0000400, 0.0000400
5: 0.0012767, 0.0017626, 0.0012767, 0.0017626, -0.0001614, 0.0001614
6: -0.0025250, -0.0024993, -0.0025250, -0.0024993, -0.0000102, 0.0000102
7: -0.0087369, -0.0076291, -0.0087369, -0.0076291, -0.0003987, 0.0003987
8: -0.0044047, -0.0030862, -0.0044047, -0.0030862, -0.0004117, 0.0004117
9: -0.0026624, -0.0020340, -0.0026624, -0.0020340, -0.0001922, 0.0001922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000495
time: 0.50 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000507
time: 0.49 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.73 seconds
IS_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000485
IS_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000503
IS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000483, upper bound: 0.0000512
IS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000512
IS_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000495, upper bound: 0.0000501
IS_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000495, upper bound: 0.0000516
IS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000479, upper bound: 0.0000524
IS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000495, upper bound: 0.0000524
IS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000483
IS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000501
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000499
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000511
IS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000495, upper bound: 0.0000501
IS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000495, upper bound: 0.0000516
IS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000495, upper bound: 0.0000501
IS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000495, upper bound: 0.0000524
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000497
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000497
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000507
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000507
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000481
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000497
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000507
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000507
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000479
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000495
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000501, upper bound: 0.0000507
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000507
IS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000479
IS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000495
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000495
IS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000507

## BFS IS instance: IS_A1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0034555, -0.0028524, -0.0034548, -0.0028444, -0.0001736, 0.0001793
1: -0.0045187, -0.0044039, -0.0045186, -0.0044031, -0.0000313, 0.0000309
2: 0.0101705, 0.0109409, 0.0101714, 0.0109506, -0.0002189, 0.0002251
3: 1.0087343, 1.0089188, 1.0087337, 1.0089197, -0.0000553, 0.0000568
4: -0.0034033, -0.0032836, -0.0034047, -0.0032838, -0.0000342, 0.0000335
5: 0.0013061, 0.0017676, 0.0013066, 0.0017737, -0.0001326, 0.0001368
6: -0.0025217, -0.0024987, -0.0025218, -0.0024983, -0.0000076, 0.0000082
7: -0.0087586, -0.0077167, -0.0087729, -0.0077206, -0.0003317, 0.0003157
8: -0.0044066, -0.0031541, -0.0044208, -0.0031557, -0.0003534, 0.0003471
9: -0.0026312, -0.0020365, -0.0026302, -0.0020302, -0.0001629, 0.0001648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000425, upper bound: 0.0000409
time: 0.47 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000425, upper bound: 0.0000409
time: 0.46 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0034548, -0.0028486, -0.0034549, -0.0028424, -0.0001990, 0.0001627
1: -0.0045186, -0.0044035, -0.0045186, -0.0044029, -0.0000335, 0.0000298
2: 0.0101715, 0.0109455, 0.0101714, 0.0109530, -0.0002496, 0.0002055
3: 1.0087337, 1.0089194, 1.0087337, 1.0089198, -0.0000580, 0.0000547
4: -0.0034039, -0.0032838, -0.0034050, -0.0032838, -0.0000315, 0.0000379
5: 0.0013067, 0.0017705, 0.0013066, 0.0017752, -0.0001519, 0.0001243
6: -0.0025218, -0.0024985, -0.0025218, -0.0024981, -0.0000092, 0.0000070
7: -0.0087635, -0.0077207, -0.0087762, -0.0077206, -0.0002929, 0.0003702
8: -0.0044131, -0.0031558, -0.0044242, -0.0031556, -0.0003265, 0.0003913
9: -0.0026301, -0.0020337, -0.0026302, -0.0020289, -0.0001824, 0.0001531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000425, upper bound: 0.0000425
time: 0.46 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000425, upper bound: 0.0000425
time: 0.46 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0034838, -0.0028649, -0.0034555, -0.0028524, -0.0002327, 0.0001817
1: -0.0045193, -0.0044033, -0.0045187, -0.0044039, -0.0000318, 0.0000312
2: 0.0101387, 0.0109272, 0.0101705, 0.0109409, -0.0002852, 0.0002280
3: 1.0087209, 1.0089133, 1.0087343, 1.0089188, -0.0000822, 0.0000577
4: -0.0034015, -0.0032795, -0.0034033, -0.0032836, -0.0000347, 0.0000421
5: 0.0012849, 0.0017582, 0.0013061, 0.0017676, -0.0001771, 0.0001387
6: -0.0025250, -0.0024997, -0.0025217, -0.0024987, -0.0000136, 0.0000083
7: -0.0087258, -0.0076436, -0.0087586, -0.0077167, -0.0003366, 0.0004709
8: -0.0043908, -0.0031163, -0.0044066, -0.0031541, -0.0003578, 0.0004255
9: -0.0026464, -0.0020422, -0.0026312, -0.0020365, -0.0001930, 0.0001668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000406, upper bound: 0.0000433
time: 0.47 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000405, upper bound: 0.0000433
time: 0.49 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0034838, -0.0028627, -0.0034548, -0.0028486, -0.0002263, 0.0002028
1: -0.0045193, -0.0044031, -0.0045186, -0.0044035, -0.0000310, 0.0000333
2: 0.0101387, 0.0109298, 0.0101715, 0.0109455, -0.0002773, 0.0002534
3: 1.0087209, 1.0089135, 1.0087337, 1.0089194, -0.0000821, 0.0000601
4: -0.0034019, -0.0032795, -0.0034039, -0.0032838, -0.0000383, 0.0000409
5: 0.0012848, 0.0017599, 0.0013067, 0.0017705, -0.0001722, 0.0001547
6: -0.0025250, -0.0024995, -0.0025218, -0.0024985, -0.0000134, 0.0000098
7: -0.0087277, -0.0076436, -0.0087635, -0.0077207, -0.0003825, 0.0004581
8: -0.0043947, -0.0031163, -0.0044131, -0.0031558, -0.0003940, 0.0004133
9: -0.0026464, -0.0020404, -0.0026301, -0.0020337, -0.0001871, 0.0001829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000421, upper bound: 0.0000433
time: 0.47 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000421, upper bound: 0.0000433
time: 0.49 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0034555, -0.0028524, -0.0034672, -0.0028429, -0.0002040, 0.0002127
1: -0.0045187, -0.0044039, -0.0045224, -0.0044020, -0.0000400, 0.0000405
2: 0.0101705, 0.0109409, 0.0101545, 0.0109531, -0.0002606, 0.0002705
3: 1.0087343, 1.0089188, 1.0087334, 1.0089242, -0.0000691, 0.0000681
4: -0.0034033, -0.0032836, -0.0034052, -0.0032810, -0.0000418, 0.0000405
5: 0.0013061, 0.0017676, 0.0012971, 0.0017749, -0.0001561, 0.0001626
6: -0.0025217, -0.0024987, -0.0025218, -0.0024983, -0.0000076, 0.0000083
7: -0.0087586, -0.0077167, -0.0087690, -0.0077114, -0.0003735, 0.0003501
8: -0.0044066, -0.0031541, -0.0044277, -0.0031244, -0.0004365, 0.0004247
9: -0.0026312, -0.0020365, -0.0026471, -0.0020258, -0.0002020, 0.0002065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000419, upper bound: 0.0000426
time: 0.47 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000417, upper bound: 0.0000426
time: 0.48 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0034548, -0.0028486, -0.0034672, -0.0028410, -0.0002239, 0.0002108
1: -0.0045186, -0.0044035, -0.0045224, -0.0044018, -0.0000413, 0.0000406
2: 0.0101715, 0.0109455, 0.0101545, 0.0109554, -0.0002843, 0.0002688
3: 1.0087337, 1.0089194, 1.0087334, 1.0089246, -0.0000716, 0.0000681
4: -0.0034039, -0.0032838, -0.0034055, -0.0032810, -0.0000416, 0.0000438
5: 0.0013067, 0.0017705, 0.0012970, 0.0017763, -0.0001712, 0.0001613
6: -0.0025218, -0.0024985, -0.0025218, -0.0024982, -0.0000093, 0.0000074
7: -0.0087635, -0.0077207, -0.0087713, -0.0077114, -0.0003633, 0.0003965
8: -0.0044131, -0.0031558, -0.0044307, -0.0031244, -0.0004352, 0.0004571
9: -0.0026301, -0.0020337, -0.0026471, -0.0020244, -0.0002160, 0.0002060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000419, upper bound: 0.0000439
time: 0.48 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000417, upper bound: 0.0000439
time: 0.48 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0034838, -0.0028649, -0.0034687, -0.0028503, -0.0002661, 0.0002300
1: -0.0045193, -0.0044033, -0.0045224, -0.0044029, -0.0000412, 0.0000420
2: 0.0101387, 0.0109272, 0.0101533, 0.0109436, -0.0003310, 0.0002916
3: 1.0087209, 1.0089133, 1.0087337, 1.0089234, -0.0000960, 0.0000711
4: -0.0034015, -0.0032795, -0.0034038, -0.0032807, -0.0000449, 0.0000498
5: 0.0012849, 0.0017582, 0.0012960, 0.0017692, -0.0002029, 0.0001758
6: -0.0025250, -0.0024997, -0.0025217, -0.0024987, -0.0000137, 0.0000087
7: -0.0087258, -0.0076436, -0.0087554, -0.0077031, -0.0004072, 0.0005106
8: -0.0043908, -0.0031163, -0.0044140, -0.0031218, -0.0004668, 0.0005101
9: -0.0026464, -0.0020422, -0.0026479, -0.0020321, -0.0002357, 0.0002199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000402, upper bound: 0.0000448
time: 0.48 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000397, upper bound: 0.0000448
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0034838, -0.0028627, -0.0034672, -0.0028473, -0.0002598, 0.0002365
1: -0.0045193, -0.0044031, -0.0045224, -0.0044024, -0.0000404, 0.0000430
2: 0.0101387, 0.0109298, 0.0101546, 0.0109477, -0.0003233, 0.0002991
3: 1.0087209, 1.0089135, 1.0087334, 1.0089240, -0.0000959, 0.0000714
4: -0.0034019, -0.0032795, -0.0034044, -0.0032810, -0.0000460, 0.0000486
5: 0.0012848, 0.0017599, 0.0012971, 0.0017715, -0.0001981, 0.0001807
6: -0.0025250, -0.0024995, -0.0025218, -0.0024985, -0.0000134, 0.0000099
7: -0.0087277, -0.0076436, -0.0087572, -0.0077115, -0.0004246, 0.0004980
8: -0.0043947, -0.0031163, -0.0044193, -0.0031245, -0.0004778, 0.0004981
9: -0.0026464, -0.0020404, -0.0026470, -0.0020294, -0.0002299, 0.0002249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000418, upper bound: 0.0000448
time: 0.50 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000410, upper bound: 0.0000448
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0034555, -0.0028524, -0.0034838, -0.0028649, -0.0001817, 0.0002327
1: -0.0045187, -0.0044039, -0.0045193, -0.0044033, -0.0000312, 0.0000318
2: 0.0101705, 0.0109409, 0.0101387, 0.0109272, -0.0002280, 0.0002852
3: 1.0087343, 1.0089188, 1.0087209, 1.0089133, -0.0000577, 0.0000822
4: -0.0034033, -0.0032836, -0.0034015, -0.0032795, -0.0000421, 0.0000347
5: 0.0013061, 0.0017676, 0.0012849, 0.0017582, -0.0001387, 0.0001771
6: -0.0025217, -0.0024987, -0.0025250, -0.0024997, -0.0000083, 0.0000136
7: -0.0087586, -0.0077167, -0.0087258, -0.0076436, -0.0004709, 0.0003366
8: -0.0044066, -0.0031541, -0.0043908, -0.0031163, -0.0004255, 0.0003578
9: -0.0026312, -0.0020365, -0.0026464, -0.0020422, -0.0001668, 0.0001930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000433, upper bound: 0.0000406
time: 0.47 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000433, upper bound: 0.0000405
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0034548, -0.0028486, -0.0034838, -0.0028627, -0.0002028, 0.0002263
1: -0.0045186, -0.0044035, -0.0045193, -0.0044031, -0.0000333, 0.0000310
2: 0.0101715, 0.0109455, 0.0101387, 0.0109298, -0.0002534, 0.0002773
3: 1.0087337, 1.0089194, 1.0087209, 1.0089135, -0.0000601, 0.0000821
4: -0.0034039, -0.0032838, -0.0034019, -0.0032795, -0.0000409, 0.0000383
5: 0.0013067, 0.0017705, 0.0012848, 0.0017599, -0.0001547, 0.0001722
6: -0.0025218, -0.0024985, -0.0025250, -0.0024995, -0.0000098, 0.0000134
7: -0.0087635, -0.0077207, -0.0087277, -0.0076436, -0.0004581, 0.0003825
8: -0.0044131, -0.0031558, -0.0043947, -0.0031163, -0.0004133, 0.0003940
9: -0.0026301, -0.0020337, -0.0026464, -0.0020404, -0.0001829, 0.0001871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000433, upper bound: 0.0000421
time: 0.48 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000433, upper bound: 0.0000421
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0034841, -0.0028727, -0.0034838, -0.0028649, -0.0001809, 0.0001858
1: -0.0045194, -0.0044041, -0.0045193, -0.0044033, -0.0000322, 0.0000316
2: 0.0101385, 0.0109175, 0.0101387, 0.0109272, -0.0002274, 0.0002324
3: 1.0087211, 1.0089124, 1.0087209, 1.0089133, -0.0000590, 0.0000607
4: -0.0034001, -0.0032795, -0.0034015, -0.0032795, -0.0000352, 0.0000346
5: 0.0012846, 0.0017523, 0.0012849, 0.0017582, -0.0001381, 0.0001418
6: -0.0025250, -0.0025001, -0.0025250, -0.0024997, -0.0000083, 0.0000090
7: -0.0087129, -0.0076371, -0.0087258, -0.0076436, -0.0003501, 0.0003347
8: -0.0043762, -0.0031160, -0.0043908, -0.0031163, -0.0003622, 0.0003579
9: -0.0026463, -0.0020487, -0.0026464, -0.0020422, -0.0001674, 0.0001684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000424, upper bound: 0.0000421
time: 0.48 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000421, upper bound: 0.0000421
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0034837, -0.0028687, -0.0034838, -0.0028627, -0.0002048, 0.0001702
1: -0.0045192, -0.0044038, -0.0045193, -0.0044031, -0.0000342, 0.0000307
2: 0.0101388, 0.0109225, 0.0101387, 0.0109298, -0.0002560, 0.0002143
3: 1.0087209, 1.0089128, 1.0087209, 1.0089135, -0.0000615, 0.0000582
4: -0.0034008, -0.0032795, -0.0034019, -0.0032795, -0.0000327, 0.0000387
5: 0.0012849, 0.0017553, 0.0012848, 0.0017599, -0.0001562, 0.0001300
6: -0.0025250, -0.0024999, -0.0025250, -0.0024995, -0.0000099, 0.0000078
7: -0.0087154, -0.0076436, -0.0087277, -0.0076436, -0.0003112, 0.0003872
8: -0.0043840, -0.0031164, -0.0043947, -0.0031163, -0.0003382, 0.0003987
9: -0.0026464, -0.0020451, -0.0026464, -0.0020404, -0.0001855, 0.0001583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000421, upper bound: 0.0000433
time: 0.52 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000421, upper bound: 0.0000433
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0034555, -0.0028524, -0.0034942, -0.0028645, -0.0002012, 0.0002615
1: -0.0045187, -0.0044039, -0.0045230, -0.0044022, -0.0000399, 0.0000417
2: 0.0101705, 0.0109409, 0.0101243, 0.0109278, -0.0002565, 0.0003253
3: 1.0087343, 1.0089188, 1.0087214, 1.0089172, -0.0000661, 0.0000849
4: -0.0034033, -0.0032836, -0.0034017, -0.0032770, -0.0000489, 0.0000398
5: 0.0013061, 0.0017676, 0.0012768, 0.0017586, -0.0001539, 0.0001993
6: -0.0025217, -0.0024987, -0.0025250, -0.0024996, -0.0000084, 0.0000137
7: -0.0087586, -0.0077167, -0.0087273, -0.0076292, -0.0005037, 0.0003536
8: -0.0044066, -0.0031541, -0.0043946, -0.0030863, -0.0005017, 0.0004158
9: -0.0026312, -0.0020365, -0.0026623, -0.0020384, -0.0001976, 0.0002318

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000429, upper bound: 0.0000424
time: 0.48 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000429, upper bound: 0.0000424
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0034548, -0.0028486, -0.0034942, -0.0028628, -0.0002212, 0.0002588
1: -0.0045186, -0.0044035, -0.0045230, -0.0044020, -0.0000412, 0.0000417
2: 0.0101715, 0.0109455, 0.0101243, 0.0109302, -0.0002805, 0.0003220
3: 1.0087337, 1.0089194, 1.0087214, 1.0089177, -0.0000684, 0.0000852
4: -0.0034039, -0.0032838, -0.0034021, -0.0032770, -0.0000484, 0.0000432
5: 0.0013067, 0.0017705, 0.0012768, 0.0017598, -0.0001691, 0.0001973
6: -0.0025218, -0.0024985, -0.0025250, -0.0024995, -0.0000099, 0.0000134
7: -0.0087635, -0.0077207, -0.0087283, -0.0076292, -0.0004951, 0.0003984
8: -0.0044131, -0.0031558, -0.0043986, -0.0030863, -0.0004967, 0.0004498
9: -0.0026301, -0.0020337, -0.0026624, -0.0020367, -0.0002123, 0.0002300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000429, upper bound: 0.0000438
time: 0.51 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000429, upper bound: 0.0000438
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0034841, -0.0028727, -0.0034942, -0.0028645, -0.0002101, 0.0002197
1: -0.0045194, -0.0044041, -0.0045230, -0.0044022, -0.0000409, 0.0000413
2: 0.0101385, 0.0109175, 0.0101243, 0.0109278, -0.0002678, 0.0002784
3: 1.0087211, 1.0089124, 1.0087214, 1.0089172, -0.0000730, 0.0000719
4: -0.0034001, -0.0032795, -0.0034017, -0.0032770, -0.0000428, 0.0000415
5: 0.0012846, 0.0017523, 0.0012768, 0.0017586, -0.0001607, 0.0001679
6: -0.0025250, -0.0025001, -0.0025250, -0.0024996, -0.0000084, 0.0000091
7: -0.0087129, -0.0076371, -0.0087273, -0.0076292, -0.0003929, 0.0003666
8: -0.0043762, -0.0031160, -0.0043946, -0.0030863, -0.0004462, 0.0004341
9: -0.0026463, -0.0020487, -0.0026623, -0.0020384, -0.0002062, 0.0002105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000418, upper bound: 0.0000436
time: 0.51 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000410, upper bound: 0.0000436
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0034837, -0.0028687, -0.0034942, -0.0028628, -0.0002284, 0.0002176
1: -0.0045192, -0.0044038, -0.0045230, -0.0044020, -0.0000420, 0.0000414
2: 0.0101388, 0.0109225, 0.0101243, 0.0109302, -0.0002894, 0.0002766
3: 1.0087209, 1.0089128, 1.0087214, 1.0089177, -0.0000755, 0.0000716
4: -0.0034008, -0.0032795, -0.0034021, -0.0032770, -0.0000427, 0.0000445
5: 0.0012849, 0.0017553, 0.0012768, 0.0017598, -0.0001746, 0.0001664
6: -0.0025250, -0.0024999, -0.0025250, -0.0024995, -0.0000100, 0.0000081
7: -0.0087154, -0.0076436, -0.0087283, -0.0076292, -0.0003805, 0.0004110
8: -0.0043840, -0.0031164, -0.0043986, -0.0030863, -0.0004451, 0.0004634
9: -0.0026464, -0.0020451, -0.0026624, -0.0020367, -0.0002188, 0.0002103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000418, upper bound: 0.0000448
time: 0.50 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000410, upper bound: 0.0000448
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0034672, -0.0028429, -0.0034555, -0.0028524, -0.0002127, 0.0002040
1: -0.0045224, -0.0044020, -0.0045187, -0.0044039, -0.0000405, 0.0000400
2: 0.0101545, 0.0109531, 0.0101705, 0.0109409, -0.0002705, 0.0002606
3: 1.0087334, 1.0089242, 1.0087343, 1.0089188, -0.0000681, 0.0000691
4: -0.0034052, -0.0032810, -0.0034033, -0.0032836, -0.0000405, 0.0000418
5: 0.0012971, 0.0017749, 0.0013061, 0.0017676, -0.0001626, 0.0001561
6: -0.0025218, -0.0024983, -0.0025217, -0.0024987, -0.0000083, 0.0000076
7: -0.0087690, -0.0077114, -0.0087586, -0.0077167, -0.0003501, 0.0003735
8: -0.0044277, -0.0031244, -0.0044066, -0.0031541, -0.0004247, 0.0004365
9: -0.0026471, -0.0020258, -0.0026312, -0.0020365, -0.0002065, 0.0002020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000426, upper bound: 0.0000419
time: 0.47 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000426, upper bound: 0.0000417
time: 0.50 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0034672, -0.0028410, -0.0034548, -0.0028486, -0.0002108, 0.0002239
1: -0.0045224, -0.0044018, -0.0045186, -0.0044035, -0.0000406, 0.0000413
2: 0.0101545, 0.0109554, 0.0101715, 0.0109455, -0.0002688, 0.0002843
3: 1.0087334, 1.0089246, 1.0087337, 1.0089194, -0.0000681, 0.0000716
4: -0.0034055, -0.0032810, -0.0034039, -0.0032838, -0.0000438, 0.0000416
5: 0.0012970, 0.0017763, 0.0013067, 0.0017705, -0.0001613, 0.0001712
6: -0.0025218, -0.0024982, -0.0025218, -0.0024985, -0.0000074, 0.0000093
7: -0.0087713, -0.0077114, -0.0087635, -0.0077207, -0.0003965, 0.0003633
8: -0.0044307, -0.0031244, -0.0044131, -0.0031558, -0.0004571, 0.0004352
9: -0.0026471, -0.0020244, -0.0026301, -0.0020337, -0.0002060, 0.0002160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_B1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000439, upper bound: 0.0000419
time: 0.49 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000439, upper bound: 0.0000417
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0034942, -0.0028645, -0.0034555, -0.0028524, -0.0002615, 0.0002012
1: -0.0045230, -0.0044022, -0.0045187, -0.0044039, -0.0000417, 0.0000399
2: 0.0101243, 0.0109278, 0.0101705, 0.0109409, -0.0003253, 0.0002565
3: 1.0087214, 1.0089172, 1.0087343, 1.0089188, -0.0000849, 0.0000661
4: -0.0034017, -0.0032770, -0.0034033, -0.0032836, -0.0000398, 0.0000489
5: 0.0012768, 0.0017586, 0.0013061, 0.0017676, -0.0001993, 0.0001539
6: -0.0025250, -0.0024996, -0.0025217, -0.0024987, -0.0000137, 0.0000084
7: -0.0087273, -0.0076292, -0.0087586, -0.0077167, -0.0003536, 0.0005037
8: -0.0043946, -0.0030863, -0.0044066, -0.0031541, -0.0004158, 0.0005017
9: -0.0026623, -0.0020384, -0.0026312, -0.0020365, -0.0002318, 0.0001976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_B1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000424, upper bound: 0.0000429
time: 0.48 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000424, upper bound: 0.0000429
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0034942, -0.0028628, -0.0034548, -0.0028486, -0.0002588, 0.0002212
1: -0.0045230, -0.0044020, -0.0045186, -0.0044035, -0.0000417, 0.0000412
2: 0.0101243, 0.0109302, 0.0101715, 0.0109455, -0.0003220, 0.0002805
3: 1.0087214, 1.0089177, 1.0087337, 1.0089194, -0.0000852, 0.0000684
4: -0.0034021, -0.0032770, -0.0034039, -0.0032838, -0.0000432, 0.0000484
5: 0.0012768, 0.0017598, 0.0013067, 0.0017705, -0.0001973, 0.0001691
6: -0.0025250, -0.0024995, -0.0025218, -0.0024985, -0.0000134, 0.0000099
7: -0.0087283, -0.0076292, -0.0087635, -0.0077207, -0.0003984, 0.0004951
8: -0.0043986, -0.0030863, -0.0044131, -0.0031558, -0.0004498, 0.0004967
9: -0.0026624, -0.0020367, -0.0026301, -0.0020337, -0.0002300, 0.0002123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000429
time: 0.49 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000429
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0034687, -0.0028503, -0.0034672, -0.0028429, -0.0001789, 0.0001846
1: -0.0045224, -0.0044029, -0.0045224, -0.0044020, -0.0000326, 0.0000321
2: 0.0101533, 0.0109436, 0.0101545, 0.0109531, -0.0002256, 0.0002318
3: 1.0087337, 1.0089234, 1.0087334, 1.0089242, -0.0000574, 0.0000590
4: -0.0034038, -0.0032807, -0.0034052, -0.0032810, -0.0000353, 0.0000346
5: 0.0012960, 0.0017692, 0.0012971, 0.0017749, -0.0001366, 0.0001409
6: -0.0025217, -0.0024987, -0.0025218, -0.0024983, -0.0000077, 0.0000084
7: -0.0087554, -0.0077031, -0.0087690, -0.0077114, -0.0003377, 0.0003214
8: -0.0044140, -0.0031218, -0.0044277, -0.0031244, -0.0003652, 0.0003590
9: -0.0026479, -0.0020321, -0.0026471, -0.0020258, -0.0001687, 0.0001706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A2_B1_B2_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000437, upper bound: 0.0000404
time: 0.48 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000439, upper bound: 0.0000404
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0034672, -0.0028473, -0.0034672, -0.0028410, -0.0002041, 0.0001675
1: -0.0045224, -0.0044024, -0.0045224, -0.0044018, -0.0000347, 0.0000311
2: 0.0101546, 0.0109477, 0.0101545, 0.0109554, -0.0002561, 0.0002118
3: 1.0087334, 1.0089240, 1.0087334, 1.0089246, -0.0000600, 0.0000567
4: -0.0034044, -0.0032810, -0.0034055, -0.0032810, -0.0000325, 0.0000390
5: 0.0012971, 0.0017715, 0.0012970, 0.0017763, -0.0001558, 0.0001280
6: -0.0025218, -0.0024985, -0.0025218, -0.0024982, -0.0000093, 0.0000071
7: -0.0087572, -0.0077115, -0.0087713, -0.0077114, -0.0002969, 0.0003760
8: -0.0044193, -0.0031245, -0.0044307, -0.0031244, -0.0003378, 0.0004029
9: -0.0026470, -0.0020294, -0.0026471, -0.0020244, -0.0001881, 0.0001590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_B1_B2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000439, upper bound: 0.0000419
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000439, upper bound: 0.0000417
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0034942, -0.0028645, -0.0034687, -0.0028503, -0.0002381, 0.0001869
1: -0.0045230, -0.0044022, -0.0045224, -0.0044029, -0.0000331, 0.0000325
2: 0.0101243, 0.0109278, 0.0101533, 0.0109436, -0.0002920, 0.0002347
3: 1.0087214, 1.0089172, 1.0087337, 1.0089234, -0.0000843, 0.0000597
4: -0.0034017, -0.0032770, -0.0034038, -0.0032807, -0.0000358, 0.0000432
5: 0.0012768, 0.0017586, 0.0012960, 0.0017692, -0.0001811, 0.0001427
6: -0.0025250, -0.0024996, -0.0025217, -0.0024987, -0.0000137, 0.0000085
7: -0.0087273, -0.0076292, -0.0087554, -0.0077031, -0.0003424, 0.0004770
8: -0.0043946, -0.0030863, -0.0044140, -0.0031218, -0.0003696, 0.0004375
9: -0.0026623, -0.0020384, -0.0026479, -0.0020321, -0.0001988, 0.0001726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A2_B1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000421, upper bound: 0.0000429
time: 0.54 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000424, upper bound: 0.0000429
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0034942, -0.0028628, -0.0034672, -0.0028473, -0.0002311, 0.0002079
1: -0.0045230, -0.0044020, -0.0045224, -0.0044024, -0.0000323, 0.0000346
2: 0.0101243, 0.0109302, 0.0101546, 0.0109477, -0.0002836, 0.0002598
3: 1.0087214, 1.0089177, 1.0087334, 1.0089240, -0.0000841, 0.0000622
4: -0.0034021, -0.0032770, -0.0034044, -0.0032810, -0.0000394, 0.0000419
5: 0.0012768, 0.0017598, 0.0012971, 0.0017715, -0.0001759, 0.0001586
6: -0.0025250, -0.0024995, -0.0025218, -0.0024985, -0.0000135, 0.0000099
7: -0.0087283, -0.0076292, -0.0087572, -0.0077115, -0.0003881, 0.0004620
8: -0.0043986, -0.0030863, -0.0044193, -0.0031245, -0.0004054, 0.0004244
9: -0.0026624, -0.0020367, -0.0026470, -0.0020294, -0.0001929, 0.0001885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000429
time: 0.51 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000429
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0034687, -0.0028503, -0.0034838, -0.0028649, -0.0002300, 0.0002661
1: -0.0045224, -0.0044029, -0.0045193, -0.0044033, -0.0000420, 0.0000412
2: 0.0101533, 0.0109436, 0.0101387, 0.0109272, -0.0002916, 0.0003310
3: 1.0087337, 1.0089234, 1.0087209, 1.0089133, -0.0000711, 0.0000960
4: -0.0034038, -0.0032807, -0.0034015, -0.0032795, -0.0000498, 0.0000449
5: 0.0012960, 0.0017692, 0.0012849, 0.0017582, -0.0001758, 0.0002029
6: -0.0025217, -0.0024987, -0.0025250, -0.0024997, -0.0000087, 0.0000137
7: -0.0087554, -0.0077031, -0.0087258, -0.0076436, -0.0005106, 0.0004072
8: -0.0044140, -0.0031218, -0.0043908, -0.0031163, -0.0005101, 0.0004668
9: -0.0026479, -0.0020321, -0.0026464, -0.0020422, -0.0002199, 0.0002357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000448, upper bound: 0.0000402
time: 0.49 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000448, upper bound: 0.0000397
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0034672, -0.0028473, -0.0034838, -0.0028627, -0.0002365, 0.0002598
1: -0.0045224, -0.0044024, -0.0045193, -0.0044031, -0.0000430, 0.0000404
2: 0.0101546, 0.0109477, 0.0101387, 0.0109298, -0.0002991, 0.0003233
3: 1.0087334, 1.0089240, 1.0087209, 1.0089135, -0.0000714, 0.0000959
4: -0.0034044, -0.0032810, -0.0034019, -0.0032795, -0.0000486, 0.0000460
5: 0.0012971, 0.0017715, 0.0012848, 0.0017599, -0.0001807, 0.0001981
6: -0.0025218, -0.0024985, -0.0025250, -0.0024995, -0.0000099, 0.0000134
7: -0.0087572, -0.0077115, -0.0087277, -0.0076436, -0.0004980, 0.0004246
8: -0.0044193, -0.0031245, -0.0043947, -0.0031163, -0.0004981, 0.0004778
9: -0.0026470, -0.0020294, -0.0026464, -0.0020404, -0.0002249, 0.0002299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_B2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000448, upper bound: 0.0000418
time: 0.49 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000448, upper bound: 0.0000410
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0034942, -0.0028645, -0.0034841, -0.0028727, -0.0002197, 0.0002101
1: -0.0045230, -0.0044022, -0.0045194, -0.0044041, -0.0000413, 0.0000409
2: 0.0101243, 0.0109278, 0.0101385, 0.0109175, -0.0002784, 0.0002678
3: 1.0087214, 1.0089172, 1.0087211, 1.0089124, -0.0000719, 0.0000730
4: -0.0034017, -0.0032770, -0.0034001, -0.0032795, -0.0000415, 0.0000428
5: 0.0012768, 0.0017586, 0.0012846, 0.0017523, -0.0001679, 0.0001607
6: -0.0025250, -0.0024996, -0.0025250, -0.0025001, -0.0000091, 0.0000084
7: -0.0087273, -0.0076292, -0.0087129, -0.0076371, -0.0003666, 0.0003929
8: -0.0043946, -0.0030863, -0.0043762, -0.0031160, -0.0004341, 0.0004462
9: -0.0026623, -0.0020384, -0.0026463, -0.0020487, -0.0002105, 0.0002062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_B2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000424, upper bound: 0.0000429
time: 0.53 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000424, upper bound: 0.0000429
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0034942, -0.0028628, -0.0034837, -0.0028687, -0.0002176, 0.0002284
1: -0.0045230, -0.0044020, -0.0045192, -0.0044038, -0.0000414, 0.0000420
2: 0.0101243, 0.0109302, 0.0101388, 0.0109225, -0.0002766, 0.0002894
3: 1.0087214, 1.0089177, 1.0087209, 1.0089128, -0.0000716, 0.0000755
4: -0.0034021, -0.0032770, -0.0034008, -0.0032795, -0.0000445, 0.0000427
5: 0.0012768, 0.0017598, 0.0012849, 0.0017553, -0.0001664, 0.0001746
6: -0.0025250, -0.0024995, -0.0025250, -0.0024999, -0.0000081, 0.0000100
7: -0.0087283, -0.0076292, -0.0087154, -0.0076436, -0.0004110, 0.0003805
8: -0.0043986, -0.0030863, -0.0043840, -0.0031164, -0.0004634, 0.0004451
9: -0.0026624, -0.0020367, -0.0026464, -0.0020451, -0.0002103, 0.0002188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_B2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000429
time: 0.49 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000429
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0034687, -0.0028503, -0.0034942, -0.0028645, -0.0001869, 0.0002381
1: -0.0045224, -0.0044029, -0.0045230, -0.0044022, -0.0000325, 0.0000331
2: 0.0101533, 0.0109436, 0.0101243, 0.0109278, -0.0002347, 0.0002920
3: 1.0087337, 1.0089234, 1.0087214, 1.0089172, -0.0000597, 0.0000843
4: -0.0034038, -0.0032807, -0.0034017, -0.0032770, -0.0000432, 0.0000358
5: 0.0012960, 0.0017692, 0.0012768, 0.0017586, -0.0001427, 0.0001811
6: -0.0025217, -0.0024987, -0.0025250, -0.0024996, -0.0000085, 0.0000137
7: -0.0087554, -0.0077031, -0.0087273, -0.0076292, -0.0004770, 0.0003424
8: -0.0044140, -0.0031218, -0.0043946, -0.0030863, -0.0004375, 0.0003696
9: -0.0026479, -0.0020321, -0.0026623, -0.0020384, -0.0001726, 0.0001988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_B2_B2_A1_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000448, upper bound: 0.0000402
time: 0.51 seconds

## Relational analysis of IS_A2_B2_B2_A1_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000448, upper bound: 0.0000397
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0034672, -0.0028473, -0.0034942, -0.0028628, -0.0002079, 0.0002311
1: -0.0045224, -0.0044024, -0.0045230, -0.0044020, -0.0000346, 0.0000323
2: 0.0101546, 0.0109477, 0.0101243, 0.0109302, -0.0002598, 0.0002836
3: 1.0087334, 1.0089240, 1.0087214, 1.0089177, -0.0000622, 0.0000841
4: -0.0034044, -0.0032810, -0.0034021, -0.0032770, -0.0000419, 0.0000394
5: 0.0012971, 0.0017715, 0.0012768, 0.0017598, -0.0001586, 0.0001759
6: -0.0025218, -0.0024985, -0.0025250, -0.0024995, -0.0000099, 0.0000135
7: -0.0087572, -0.0077115, -0.0087283, -0.0076292, -0.0004620, 0.0003881
8: -0.0044193, -0.0031245, -0.0043986, -0.0030863, -0.0004244, 0.0004054
9: -0.0026470, -0.0020294, -0.0026624, -0.0020367, -0.0001885, 0.0001929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A2_B2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000444, upper bound: 0.0000410
time: 0.51 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000448, upper bound: 0.0000410
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0034964, -0.0028713, -0.0034942, -0.0028645, -0.0001863, 0.0001909
1: -0.0045231, -0.0044031, -0.0045230, -0.0044022, -0.0000335, 0.0000328
2: 0.0101229, 0.0109188, 0.0101243, 0.0109278, -0.0002339, 0.0002387
3: 1.0087216, 1.0089166, 1.0087214, 1.0089172, -0.0000612, 0.0000628
4: -0.0034003, -0.0032769, -0.0034017, -0.0032770, -0.0000361, 0.0000356
5: 0.0012752, 0.0017533, 0.0012768, 0.0017586, -0.0001422, 0.0001457
6: -0.0025250, -0.0025001, -0.0025250, -0.0024996, -0.0000085, 0.0000092
7: -0.0087138, -0.0076223, -0.0087273, -0.0076292, -0.0003581, 0.0003433
8: -0.0043801, -0.0030863, -0.0043946, -0.0030863, -0.0003727, 0.0003687
9: -0.0026626, -0.0020450, -0.0026623, -0.0020384, -0.0001731, 0.0001740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000437, upper bound: 0.0000417
time: 0.51 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000417
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0034942, -0.0028689, -0.0034942, -0.0028628, -0.0002100, 0.0001747
1: -0.0045230, -0.0044027, -0.0045230, -0.0044020, -0.0000354, 0.0000320
2: 0.0101244, 0.0109229, 0.0101243, 0.0109302, -0.0002623, 0.0002201
3: 1.0087214, 1.0089170, 1.0087214, 1.0089177, -0.0000637, 0.0000605
4: -0.0034010, -0.0032770, -0.0034021, -0.0032770, -0.0000336, 0.0000397
5: 0.0012768, 0.0017552, 0.0012768, 0.0017598, -0.0001602, 0.0001334
6: -0.0025250, -0.0024998, -0.0025250, -0.0024995, -0.0000101, 0.0000080
7: -0.0087138, -0.0076293, -0.0087283, -0.0076292, -0.0003196, 0.0003954
8: -0.0043881, -0.0030864, -0.0043986, -0.0030863, -0.0003489, 0.0004092
9: -0.0026623, -0.0020416, -0.0026624, -0.0020367, -0.0001911, 0.0001638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_B2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000429
time: 0.52 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000429
time: 0.51 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.77 seconds
IS_A1_B1_B1_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000425, upper bound: 0.0000409
IS_A1_B1_B1_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000425, upper bound: 0.0000409
IS_A1_B1_B1_A1_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000425, upper bound: 0.0000425
IS_A1_B1_B1_A1_A2_A2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000425, upper bound: 0.0000425
IS_A1_B1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000406, upper bound: 0.0000433
IS_A1_B1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000405, upper bound: 0.0000433
IS_A1_B1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000421, upper bound: 0.0000433
IS_A1_B1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000421, upper bound: 0.0000433
IS_A1_B1_B2_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000419, upper bound: 0.0000426
IS_A1_B1_B2_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000417, upper bound: 0.0000426
IS_A1_B1_B2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000419, upper bound: 0.0000439
IS_A1_B1_B2_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000417, upper bound: 0.0000439
IS_A1_B1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000402, upper bound: 0.0000448
IS_A1_B1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000397, upper bound: 0.0000448
IS_A1_B1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000418, upper bound: 0.0000448
IS_A1_B1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000410, upper bound: 0.0000448
IS_A1_B2_B1_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000433, upper bound: 0.0000406
IS_A1_B2_B1_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000433, upper bound: 0.0000405
IS_A1_B2_B1_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000433, upper bound: 0.0000421
IS_A1_B2_B1_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000433, upper bound: 0.0000421
IS_A1_B2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000424, upper bound: 0.0000421
IS_A1_B2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000421, upper bound: 0.0000421
IS_A1_B2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000421, upper bound: 0.0000433
IS_A1_B2_B1_A2_A2_A2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000421, upper bound: 0.0000433
IS_A1_B2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000429, upper bound: 0.0000424
IS_A1_B2_B2_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000429, upper bound: 0.0000424
IS_A1_B2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000429, upper bound: 0.0000438
IS_A1_B2_B2_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000429, upper bound: 0.0000438
IS_A1_B2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000418, upper bound: 0.0000436
IS_A1_B2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000410, upper bound: 0.0000436
IS_A1_B2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000418, upper bound: 0.0000448
IS_A1_B2_B2_A2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000410, upper bound: 0.0000448
IS_A2_B1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000426, upper bound: 0.0000419
IS_A2_B1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000426, upper bound: 0.0000417
IS_A2_B1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000439, upper bound: 0.0000419
IS_A2_B1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000439, upper bound: 0.0000417
IS_A2_B1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000424, upper bound: 0.0000429
IS_A2_B1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000424, upper bound: 0.0000429
IS_A2_B1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000429
IS_A2_B1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000429
IS_A2_B1_B2_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000437, upper bound: 0.0000404
IS_A2_B1_B2_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000439, upper bound: 0.0000404
IS_A2_B1_B2_A1_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000439, upper bound: 0.0000419
IS_A2_B1_B2_A1_A2_A2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000439, upper bound: 0.0000417
IS_A2_B1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000421, upper bound: 0.0000429
IS_A2_B1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000424, upper bound: 0.0000429
IS_A2_B1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000429
IS_A2_B1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000429
IS_A2_B2_B1_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000448, upper bound: 0.0000402
IS_A2_B2_B1_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000448, upper bound: 0.0000397
IS_A2_B2_B1_A1_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000448, upper bound: 0.0000418
IS_A2_B2_B1_A1_A2_A2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000448, upper bound: 0.0000410
IS_A2_B2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000424, upper bound: 0.0000429
IS_A2_B2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000424, upper bound: 0.0000429
IS_A2_B2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000429
IS_A2_B2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000429
IS_A2_B2_B2_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000448, upper bound: 0.0000402
IS_A2_B2_B2_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000448, upper bound: 0.0000397
IS_A2_B2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000444, upper bound: 0.0000410
IS_A2_B2_B2_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000448, upper bound: 0.0000410
IS_A2_B2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000437, upper bound: 0.0000417
IS_A2_B2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000417
IS_A2_B2_B2_A2_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000429
IS_A2_B2_B2_A2_A2_A2, status: Status.VERIFIED, split count: 6, time: 2.77
Output dim: 3, lower bound: -0.0000438, upper bound: 0.0000429

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.71 + 163.13 = 165.84 seconds
