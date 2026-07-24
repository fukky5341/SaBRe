## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00026658


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9880687, 0.9887469, 0.9880687, 0.9887469, -0.0004877, 0.0004877)
1: (-0.0042369, -0.0040679, -0.0042369, -0.0040679, -0.0001215, 0.0001215)
2: (0.0115040, 0.0123994, 0.0115040, 0.0123994, -0.0006440, 0.0006440)
3: (-0.0069168, -0.0065092, -0.0069168, -0.0065092, -0.0002931, 0.0002931)
4: (0.0027545, 0.0029278, 0.0027545, 0.0029278, -0.0001247, 0.0001247)
5: (0.0134284, 0.0145546, 0.0134284, 0.0145546, -0.0008100, 0.0008100)
6: (-0.0021533, -0.0018674, -0.0021533, -0.0018674, -0.0002056, 0.0002056)
7: (-0.0087088, -0.0079693, -0.0087088, -0.0079693, -0.0005319, 0.0005319)
8: (-0.0041440, -0.0037551, -0.0041440, -0.0037551, -0.0002797, 0.0002797)
9: (0.0024904, 0.0029414, 0.0024904, 0.0029414, -0.0003244, 0.0003244)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 1.30 = 2.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0003449, upper bound: 0.0003449

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003229, upper bound: 0.0002984
time: 0.46 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003228, upper bound: 0.0003229
time: 0.48 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.08 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.08
Output dim: 0, lower bound: -0.0003229, upper bound: 0.0002984
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.08
Output dim: 0, lower bound: -0.0003228, upper bound: 0.0003229

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.9880846, 0.9886815, 0.9880688, 0.9887381, -0.0004492, 0.0004270
1: -0.0042330, -0.0040842, -0.0042369, -0.0040701, -0.0001119, 0.0001064
2: 0.0115902, 0.0123785, 0.0115155, 0.0123993, -0.0005639, 0.0005932
3: -0.0069073, -0.0065485, -0.0069168, -0.0065145, -0.0002700, 0.0002566
4: 0.0027711, 0.0029237, 0.0027567, 0.0029278, -0.0001091, 0.0001148
5: 0.0135369, 0.0145283, 0.0134429, 0.0145545, -0.0007092, 0.0007461
6: -0.0021466, -0.0018950, -0.0021533, -0.0018711, -0.0001894, 0.0001800
7: -0.0086916, -0.0080405, -0.0087088, -0.0079788, -0.0004900, 0.0004657
8: -0.0041350, -0.0037926, -0.0041440, -0.0037601, -0.0002577, 0.0002449
9: 0.0025338, 0.0029308, 0.0024962, 0.0029413, -0.0002840, 0.0002988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002984, upper bound: 0.0002984
time: 0.47 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002984, upper bound: 0.0002984
time: 0.46 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.9880690, 0.9887293, 0.9880687, 0.9887469, -0.0004855, 0.0004335
1: -0.0042368, -0.0040723, -0.0042369, -0.0040679, -0.0001210, 0.0001080
2: 0.0115271, 0.0123990, 0.0115040, 0.0123994, -0.0005725, 0.0006411
3: -0.0069166, -0.0065198, -0.0069168, -0.0065092, -0.0002918, 0.0002606
4: 0.0027589, 0.0029277, 0.0027545, 0.0029278, -0.0001108, 0.0001241
5: 0.0134575, 0.0145542, 0.0134284, 0.0145546, -0.0007200, 0.0008063
6: -0.0021532, -0.0018748, -0.0021533, -0.0018674, -0.0002047, 0.0001828
7: -0.0087085, -0.0079884, -0.0087088, -0.0079693, -0.0005295, 0.0004728
8: -0.0041439, -0.0037652, -0.0041440, -0.0037551, -0.0002785, 0.0002487
9: 0.0025020, 0.0029412, 0.0024904, 0.0029414, -0.0002883, 0.0003229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002984, upper bound: 0.0003229
time: 0.45 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002984, upper bound: 0.0003229
time: 0.45 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.25 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0002984, upper bound: 0.0002984
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0002984, upper bound: 0.0002984
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0002984, upper bound: 0.0003229
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0002984, upper bound: 0.0003229

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.9880846, 0.9886815, 0.9880846, 0.9886815, -0.0003979, 0.0003979
1: -0.0042330, -0.0040842, -0.0042330, -0.0040842, -0.0000991, 0.0000991
2: 0.0115902, 0.0123785, 0.0115902, 0.0123785, -0.0005254, 0.0005254
3: -0.0069073, -0.0065485, -0.0069073, -0.0065485, -0.0002392, 0.0002392
4: 0.0027711, 0.0029237, 0.0027711, 0.0029237, -0.0001017, 0.0001017
5: 0.0135369, 0.0145283, 0.0135369, 0.0145283, -0.0006608, 0.0006608
6: -0.0021466, -0.0018950, -0.0021466, -0.0018950, -0.0001677, 0.0001677
7: -0.0086916, -0.0080405, -0.0086916, -0.0080405, -0.0004340, 0.0004340
8: -0.0041350, -0.0037926, -0.0041350, -0.0037926, -0.0002282, 0.0002282
9: 0.0025338, 0.0029308, 0.0025338, 0.0029308, -0.0002646, 0.0002646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002641, upper bound: 0.0002610
time: 0.48 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002641, upper bound: 0.0002639
time: 0.47 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.9880846, 0.9886815, 0.9880690, 0.9887293, -0.0004442, 0.0004249
1: -0.0042330, -0.0040842, -0.0042368, -0.0040723, -0.0001107, 0.0001059
2: 0.0115902, 0.0123785, 0.0115271, 0.0123990, -0.0005611, 0.0005866
3: -0.0069073, -0.0065485, -0.0069166, -0.0065198, -0.0002670, 0.0002554
4: 0.0027711, 0.0029237, 0.0027589, 0.0029277, -0.0001086, 0.0001135
5: 0.0135369, 0.0145283, 0.0134575, 0.0145542, -0.0007057, 0.0007378
6: -0.0021466, -0.0018950, -0.0021532, -0.0018748, -0.0001873, 0.0001791
7: -0.0086916, -0.0080405, -0.0087085, -0.0079884, -0.0004845, 0.0004634
8: -0.0041350, -0.0037926, -0.0041439, -0.0037652, -0.0002548, 0.0002437
9: 0.0025338, 0.0029308, 0.0025020, 0.0029412, -0.0002826, 0.0002954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002641, upper bound: 0.0002610
time: 0.47 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002641, upper bound: 0.0002639
time: 0.50 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.9880690, 0.9887293, 0.9880846, 0.9886815, -0.0004249, 0.0004442
1: -0.0042368, -0.0040723, -0.0042330, -0.0040842, -0.0001059, 0.0001107
2: 0.0115271, 0.0123990, 0.0115902, 0.0123785, -0.0005866, 0.0005611
3: -0.0069166, -0.0065198, -0.0069073, -0.0065485, -0.0002554, 0.0002670
4: 0.0027589, 0.0029277, 0.0027711, 0.0029237, -0.0001135, 0.0001086
5: 0.0134575, 0.0145542, 0.0135369, 0.0145283, -0.0007378, 0.0007057
6: -0.0021532, -0.0018748, -0.0021466, -0.0018950, -0.0001791, 0.0001873
7: -0.0087085, -0.0079884, -0.0086916, -0.0080405, -0.0004634, 0.0004845
8: -0.0041439, -0.0037652, -0.0041350, -0.0037926, -0.0002437, 0.0002548
9: 0.0025020, 0.0029412, 0.0025338, 0.0029308, -0.0002954, 0.0002826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002640, upper bound: 0.0002875
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002639, upper bound: 0.0002799
time: 0.46 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.9880690, 0.9887293, 0.9880690, 0.9887293, -0.0004325, 0.0004325
1: -0.0042368, -0.0040723, -0.0042368, -0.0040723, -0.0001078, 0.0001078
2: 0.0115271, 0.0123990, 0.0115271, 0.0123990, -0.0005712, 0.0005712
3: -0.0069166, -0.0065198, -0.0069166, -0.0065198, -0.0002600, 0.0002600
4: 0.0027589, 0.0029277, 0.0027589, 0.0029277, -0.0001105, 0.0001105
5: 0.0134575, 0.0145542, 0.0134575, 0.0145542, -0.0007184, 0.0007184
6: -0.0021532, -0.0018748, -0.0021532, -0.0018748, -0.0001823, 0.0001823
7: -0.0087085, -0.0079884, -0.0087085, -0.0079884, -0.0004718, 0.0004718
8: -0.0041439, -0.0037652, -0.0041439, -0.0037652, -0.0002481, 0.0002481
9: 0.0025020, 0.0029412, 0.0025020, 0.0029412, -0.0002877, 0.0002877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002640, upper bound: 0.0002875
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002639, upper bound: 0.0002799
time: 0.48 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.27 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -0.0002641, upper bound: 0.0002610
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -0.0002641, upper bound: 0.0002639
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -0.0002641, upper bound: 0.0002610
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -0.0002641, upper bound: 0.0002639
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -0.0002640, upper bound: 0.0002875
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -0.0002639, upper bound: 0.0002799
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -0.0002640, upper bound: 0.0002875
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -0.0002639, upper bound: 0.0002799

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9880697, 0.9887069, 0.9880846, 0.9886815, -0.0004244, 0.0004183
1: -0.0042367, -0.0040779, -0.0042330, -0.0040842, -0.0001058, 0.0001042
2: 0.0115567, 0.0123981, 0.0115902, 0.0123785, -0.0005523, 0.0005604
3: -0.0069162, -0.0065332, -0.0069073, -0.0065485, -0.0002551, 0.0002514
4: 0.0027647, 0.0029275, 0.0027711, 0.0029237, -0.0001069, 0.0001085
5: 0.0134947, 0.0145530, 0.0135369, 0.0145283, -0.0006947, 0.0007049
6: -0.0021529, -0.0018843, -0.0021466, -0.0018950, -0.0001789, 0.0001763
7: -0.0087078, -0.0080128, -0.0086916, -0.0080405, -0.0004629, 0.0004562
8: -0.0041435, -0.0037780, -0.0041350, -0.0037926, -0.0002434, 0.0002399
9: 0.0025169, 0.0029407, 0.0025338, 0.0029308, -0.0002782, 0.0002823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002585, upper bound: 0.0002788
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002585, upper bound: 0.0002788
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9879955, 0.9886315, 0.9880849, 0.9886666, -0.0005509, 0.0004253
1: -0.0042552, -0.0040967, -0.0042329, -0.0040879, -0.0001373, 0.0001060
2: 0.0116563, 0.0124962, 0.0116100, 0.0123781, -0.0005616, 0.0007275
3: -0.0069608, -0.0065786, -0.0069071, -0.0065575, -0.0003311, 0.0002556
4: 0.0027839, 0.0029465, 0.0027750, 0.0029236, -0.0001087, 0.0001408
5: 0.0136200, 0.0146764, 0.0135617, 0.0145278, -0.0007063, 0.0009150
6: -0.0021842, -0.0019161, -0.0021465, -0.0019013, -0.0002322, 0.0001793
7: -0.0087888, -0.0080951, -0.0086912, -0.0080568, -0.0006009, 0.0004638
8: -0.0041861, -0.0038213, -0.0041348, -0.0038011, -0.0003160, 0.0002439
9: 0.0025671, 0.0029901, 0.0025438, 0.0029306, -0.0002829, 0.0003664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002585, upper bound: 0.0002799
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002585, upper bound: 0.0002799
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9880697, 0.9887069, 0.9880690, 0.9887293, -0.0004322, 0.0004011
1: -0.0042367, -0.0040779, -0.0042368, -0.0040723, -0.0001077, 0.0001000
2: 0.0115567, 0.0123981, 0.0115271, 0.0123990, -0.0005297, 0.0005707
3: -0.0069162, -0.0065332, -0.0069166, -0.0065198, -0.0002598, 0.0002411
4: 0.0027647, 0.0029275, 0.0027589, 0.0029277, -0.0001025, 0.0001105
5: 0.0134947, 0.0145530, 0.0134575, 0.0145542, -0.0006662, 0.0007179
6: -0.0021529, -0.0018843, -0.0021532, -0.0018748, -0.0001822, 0.0001691
7: -0.0087078, -0.0080128, -0.0087085, -0.0079884, -0.0004714, 0.0004375
8: -0.0041435, -0.0037780, -0.0041439, -0.0037652, -0.0002479, 0.0002301
9: 0.0025169, 0.0029407, 0.0025020, 0.0029412, -0.0002668, 0.0002875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002585, upper bound: 0.0002788
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002585, upper bound: 0.0002788
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9879955, 0.9886315, 0.9880694, 0.9887131, -0.0005897, 0.0003990
1: -0.0042552, -0.0040967, -0.0042367, -0.0040764, -0.0001469, 0.0000994
2: 0.0116563, 0.0124962, 0.0115486, 0.0123986, -0.0005269, 0.0007787
3: -0.0069608, -0.0065786, -0.0069164, -0.0065295, -0.0003544, 0.0002398
4: 0.0027839, 0.0029465, 0.0027631, 0.0029276, -0.0001020, 0.0001507
5: 0.0136200, 0.0146764, 0.0134845, 0.0145536, -0.0006627, 0.0009794
6: -0.0021842, -0.0019161, -0.0021530, -0.0018817, -0.0002486, 0.0001682
7: -0.0087888, -0.0080951, -0.0087081, -0.0080061, -0.0006432, 0.0004352
8: -0.0041861, -0.0038213, -0.0041437, -0.0037745, -0.0003382, 0.0002289
9: 0.0025671, 0.0029901, 0.0025128, 0.0029409, -0.0002654, 0.0003922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002585, upper bound: 0.0002799
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002585, upper bound: 0.0002799
time: 0.48 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.44 seconds
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0002585, upper bound: 0.0002788
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0002585, upper bound: 0.0002788
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0002585, upper bound: 0.0002799
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0002585, upper bound: 0.0002799
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0002585, upper bound: 0.0002788
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0002585, upper bound: 0.0002788
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0002585, upper bound: 0.0002799
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0002585, upper bound: 0.0002799

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9880697, 0.9887069, 0.9880851, 0.9886588, -0.0003930, 0.0004179
1: -0.0042367, -0.0040779, -0.0042328, -0.0040899, -0.0000979, 0.0001041
2: 0.0115567, 0.0123981, 0.0116201, 0.0123777, -0.0005519, 0.0005190
3: -0.0069162, -0.0065332, -0.0069069, -0.0065621, -0.0002362, 0.0002512
4: 0.0027647, 0.0029275, 0.0027769, 0.0029236, -0.0001068, 0.0001004
5: 0.0134947, 0.0145530, 0.0135745, 0.0145274, -0.0006941, 0.0006527
6: -0.0021529, -0.0018843, -0.0021464, -0.0019045, -0.0001657, 0.0001762
7: -0.0087078, -0.0080128, -0.0086910, -0.0080652, -0.0004286, 0.0004558
8: -0.0041435, -0.0037780, -0.0041346, -0.0038056, -0.0002254, 0.0002397
9: 0.0025169, 0.0029407, 0.0025489, 0.0029305, -0.0002780, 0.0002614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002130, upper bound: 0.0002010
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002522, upper bound: 0.0002806
time: 0.50 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9880697, 0.9887069, 0.9879983, 0.9885958, -0.0004004, 0.0005809
1: -0.0042367, -0.0040779, -0.0042545, -0.0041056, -0.0000998, 0.0001447
2: 0.0115567, 0.0123981, 0.0117034, 0.0124924, -0.0007670, 0.0005287
3: -0.0069162, -0.0065332, -0.0069591, -0.0066000, -0.0002406, 0.0003491
4: 0.0027647, 0.0029275, 0.0027931, 0.0029458, -0.0001485, 0.0001023
5: 0.0134947, 0.0145530, 0.0136793, 0.0146716, -0.0009647, 0.0006649
6: -0.0021529, -0.0018843, -0.0021830, -0.0019311, -0.0001688, 0.0002449
7: -0.0087078, -0.0080128, -0.0087856, -0.0081340, -0.0004366, 0.0006335
8: -0.0041435, -0.0037780, -0.0041844, -0.0038417, -0.0002296, 0.0003332
9: 0.0025169, 0.0029407, 0.0025908, 0.0029882, -0.0003863, 0.0002663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002130, upper bound: 0.0002010
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002522, upper bound: 0.0002806
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9879955, 0.9886315, 0.9880851, 0.9886588, -0.0005255, 0.0004017
1: -0.0042552, -0.0040967, -0.0042328, -0.0040899, -0.0001309, 0.0001001
2: 0.0116563, 0.0124962, 0.0116201, 0.0123777, -0.0005304, 0.0006939
3: -0.0069608, -0.0065786, -0.0069069, -0.0065621, -0.0003158, 0.0002414
4: 0.0027839, 0.0029465, 0.0027769, 0.0029236, -0.0001027, 0.0001343
5: 0.0136200, 0.0146764, 0.0135745, 0.0145274, -0.0006671, 0.0008728
6: -0.0021842, -0.0019161, -0.0021464, -0.0019045, -0.0002215, 0.0001693
7: -0.0087888, -0.0080951, -0.0086910, -0.0080652, -0.0005731, 0.0004381
8: -0.0041861, -0.0038213, -0.0041346, -0.0038056, -0.0003014, 0.0002304
9: 0.0025671, 0.0029901, 0.0025489, 0.0029305, -0.0002671, 0.0003495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001857, upper bound: 0.0001658
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002520, upper bound: 0.0002738
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9879955, 0.9886315, 0.9879983, 0.9885958, -0.0004053, 0.0004305
1: -0.0042552, -0.0040967, -0.0042545, -0.0041056, -0.0001010, 0.0001073
2: 0.0116563, 0.0124962, 0.0117034, 0.0124924, -0.0005685, 0.0005353
3: -0.0069608, -0.0065786, -0.0069591, -0.0066000, -0.0002436, 0.0002588
4: 0.0027839, 0.0029465, 0.0027931, 0.0029458, -0.0001100, 0.0001036
5: 0.0136200, 0.0146764, 0.0136793, 0.0146716, -0.0007150, 0.0006732
6: -0.0021842, -0.0019161, -0.0021830, -0.0019311, -0.0001709, 0.0001815
7: -0.0087888, -0.0080951, -0.0087856, -0.0081340, -0.0004421, 0.0004695
8: -0.0041861, -0.0038213, -0.0041844, -0.0038417, -0.0002325, 0.0002469
9: 0.0025671, 0.0029901, 0.0025908, 0.0029882, -0.0002863, 0.0002696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001857, upper bound: 0.0001658
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002520, upper bound: 0.0002738
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9880697, 0.9887069, 0.9880697, 0.9887069, -0.0004008, 0.0004008
1: -0.0042367, -0.0040779, -0.0042367, -0.0040779, -0.0000999, 0.0000999
2: 0.0115567, 0.0123981, 0.0115567, 0.0123981, -0.0005293, 0.0005293
3: -0.0069162, -0.0065332, -0.0069162, -0.0065332, -0.0002409, 0.0002409
4: 0.0027647, 0.0029275, 0.0027647, 0.0029275, -0.0001024, 0.0001024
5: 0.0134947, 0.0145530, 0.0134947, 0.0145530, -0.0006657, 0.0006657
6: -0.0021529, -0.0018843, -0.0021529, -0.0018843, -0.0001690, 0.0001690
7: -0.0087078, -0.0080128, -0.0087078, -0.0080128, -0.0004371, 0.0004371
8: -0.0041435, -0.0037780, -0.0041435, -0.0037780, -0.0002299, 0.0002299
9: 0.0025169, 0.0029407, 0.0025169, 0.0029407, -0.0002666, 0.0002666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002130, upper bound: 0.0002010
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002522, upper bound: 0.0002806
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9880697, 0.9887069, 0.9879955, 0.9886315, -0.0004084, 0.0005640
1: -0.0042367, -0.0040779, -0.0042552, -0.0040967, -0.0001018, 0.0001405
2: 0.0115567, 0.0123981, 0.0116563, 0.0124962, -0.0007447, 0.0005393
3: -0.0069162, -0.0065332, -0.0069608, -0.0065786, -0.0002455, 0.0003390
4: 0.0027647, 0.0029275, 0.0027839, 0.0029465, -0.0001441, 0.0001044
5: 0.0134947, 0.0145530, 0.0136200, 0.0146764, -0.0009366, 0.0006783
6: -0.0021529, -0.0018843, -0.0021842, -0.0019161, -0.0001722, 0.0002377
7: -0.0087078, -0.0080128, -0.0087888, -0.0080951, -0.0004454, 0.0006151
8: -0.0041435, -0.0037780, -0.0041861, -0.0038213, -0.0002343, 0.0003235
9: 0.0025169, 0.0029407, 0.0025671, 0.0029901, -0.0003751, 0.0002716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002130, upper bound: 0.0002010
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002522, upper bound: 0.0002806
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9879955, 0.9886315, 0.9880697, 0.9887069, -0.0005640, 0.0004084
1: -0.0042552, -0.0040967, -0.0042367, -0.0040779, -0.0001405, 0.0001018
2: 0.0116563, 0.0124962, 0.0115567, 0.0123981, -0.0005393, 0.0007447
3: -0.0069608, -0.0065786, -0.0069162, -0.0065332, -0.0003390, 0.0002455
4: 0.0027839, 0.0029465, 0.0027647, 0.0029275, -0.0001044, 0.0001441
5: 0.0136200, 0.0146764, 0.0134947, 0.0145530, -0.0006783, 0.0009366
6: -0.0021842, -0.0019161, -0.0021529, -0.0018843, -0.0002377, 0.0001722
7: -0.0087888, -0.0080951, -0.0087078, -0.0080128, -0.0006151, 0.0004454
8: -0.0041861, -0.0038213, -0.0041435, -0.0037780, -0.0003235, 0.0002343
9: 0.0025671, 0.0029901, 0.0025169, 0.0029407, -0.0002716, 0.0003751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001857, upper bound: 0.0001657
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002520, upper bound: 0.0002738
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9879955, 0.9886315, 0.9879955, 0.9886315, -0.0004046, 0.0004046
1: -0.0042552, -0.0040967, -0.0042552, -0.0040967, -0.0001008, 0.0001008
2: 0.0116563, 0.0124962, 0.0116563, 0.0124962, -0.0005343, 0.0005343
3: -0.0069608, -0.0065786, -0.0069608, -0.0065786, -0.0002432, 0.0002432
4: 0.0027839, 0.0029465, 0.0027839, 0.0029465, -0.0001034, 0.0001034
5: 0.0136200, 0.0146764, 0.0136200, 0.0146764, -0.0006720, 0.0006720
6: -0.0021842, -0.0019161, -0.0021842, -0.0019161, -0.0001706, 0.0001706
7: -0.0087888, -0.0080951, -0.0087888, -0.0080951, -0.0004413, 0.0004413
8: -0.0041861, -0.0038213, -0.0041861, -0.0038213, -0.0002321, 0.0002321
9: 0.0025671, 0.0029901, 0.0025671, 0.0029901, -0.0002691, 0.0002691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001857, upper bound: 0.0001657
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002520, upper bound: 0.0002738
time: 0.49 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.38 seconds
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0002130, upper bound: 0.0002010
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0002522, upper bound: 0.0002806
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0002130, upper bound: 0.0002010
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0002522, upper bound: 0.0002806
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0001857, upper bound: 0.0001658
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0002520, upper bound: 0.0002738
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0001857, upper bound: 0.0001658
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0002520, upper bound: 0.0002738
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0002130, upper bound: 0.0002010
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0002522, upper bound: 0.0002806
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0002130, upper bound: 0.0002010
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0002522, upper bound: 0.0002806
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0001857, upper bound: 0.0001657
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0002520, upper bound: 0.0002738
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0001857, upper bound: 0.0001657
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0002520, upper bound: 0.0002738

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9880778, 0.9887069, 0.9880851, 0.9886588, -0.0003665, 0.0004176
1: -0.0042347, -0.0040779, -0.0042328, -0.0040899, -0.0000913, 0.0001041
2: 0.0115567, 0.0123875, 0.0116201, 0.0123777, -0.0005515, 0.0004840
3: -0.0069114, -0.0065332, -0.0069069, -0.0065621, -0.0002203, 0.0002510
4: 0.0027647, 0.0029255, 0.0027769, 0.0029236, -0.0001067, 0.0000937
5: 0.0134947, 0.0145396, 0.0135745, 0.0145274, -0.0006936, 0.0006088
6: -0.0021495, -0.0018843, -0.0021464, -0.0019045, -0.0001545, 0.0001760
7: -0.0086990, -0.0080128, -0.0086910, -0.0080652, -0.0003998, 0.0004555
8: -0.0041389, -0.0037780, -0.0041346, -0.0038056, -0.0002102, 0.0002395
9: 0.0025169, 0.0029353, 0.0025489, 0.0029305, -0.0002777, 0.0002438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002424, upper bound: 0.0002755
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002424, upper bound: 0.0002810
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9880778, 0.9887069, 0.9879983, 0.9885958, -0.0003839, 0.0005805
1: -0.0042347, -0.0040779, -0.0042545, -0.0041056, -0.0000957, 0.0001447
2: 0.0115567, 0.0123875, 0.0117034, 0.0124924, -0.0007666, 0.0005069
3: -0.0069114, -0.0065332, -0.0069591, -0.0066000, -0.0002307, 0.0003489
4: 0.0027647, 0.0029255, 0.0027931, 0.0029458, -0.0001484, 0.0000981
5: 0.0134947, 0.0145396, 0.0136793, 0.0146716, -0.0009642, 0.0006376
6: -0.0021495, -0.0018843, -0.0021830, -0.0019311, -0.0001618, 0.0002447
7: -0.0086990, -0.0080128, -0.0087856, -0.0081340, -0.0004187, 0.0006332
8: -0.0041389, -0.0037780, -0.0041844, -0.0038417, -0.0002202, 0.0003330
9: 0.0025169, 0.0029353, 0.0025908, 0.0029882, -0.0003861, 0.0002553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001923, upper bound: 0.0002365
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0001923, upper bound: 0.0002806
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9880034, 0.9886315, 0.9880851, 0.9886588, -0.0004937, 0.0004013
1: -0.0042532, -0.0040967, -0.0042328, -0.0040899, -0.0001230, 0.0001000
2: 0.0116563, 0.0124856, 0.0116201, 0.0123777, -0.0005299, 0.0006520
3: -0.0069560, -0.0065786, -0.0069069, -0.0065621, -0.0002968, 0.0002412
4: 0.0027839, 0.0029445, 0.0027769, 0.0029236, -0.0001026, 0.0001262
5: 0.0136200, 0.0146631, 0.0135745, 0.0145274, -0.0006665, 0.0008200
6: -0.0021808, -0.0019161, -0.0021464, -0.0019045, -0.0002081, 0.0001692
7: -0.0087801, -0.0080951, -0.0086910, -0.0080652, -0.0005385, 0.0004377
8: -0.0041815, -0.0038213, -0.0041346, -0.0038056, -0.0002832, 0.0002302
9: 0.0025671, 0.0029848, 0.0025489, 0.0029305, -0.0002669, 0.0003284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002043, upper bound: 0.0002253
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002043, upper bound: 0.0002738
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9880034, 0.9886315, 0.9879983, 0.9885958, -0.0003803, 0.0004302
1: -0.0042532, -0.0040967, -0.0042545, -0.0041056, -0.0000948, 0.0001072
2: 0.0116563, 0.0124856, 0.0117034, 0.0124924, -0.0005681, 0.0005022
3: -0.0069560, -0.0065786, -0.0069591, -0.0066000, -0.0002286, 0.0002586
4: 0.0027839, 0.0029445, 0.0027931, 0.0029458, -0.0001099, 0.0000972
5: 0.0136200, 0.0146631, 0.0136793, 0.0146716, -0.0007145, 0.0006316
6: -0.0021808, -0.0019161, -0.0021830, -0.0019311, -0.0001603, 0.0001813
7: -0.0087801, -0.0080951, -0.0087856, -0.0081340, -0.0004148, 0.0004692
8: -0.0041815, -0.0038213, -0.0041844, -0.0038417, -0.0002181, 0.0002467
9: 0.0025671, 0.0029848, 0.0025908, 0.0029882, -0.0002861, 0.0002529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001783, upper bound: 0.0002109
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0001783, upper bound: 0.0002738
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9880778, 0.9887069, 0.9880697, 0.9887069, -0.0003419, 0.0004002
1: -0.0042347, -0.0040779, -0.0042367, -0.0040779, -0.0000852, 0.0000997
2: 0.0115567, 0.0123875, 0.0115567, 0.0123981, -0.0005285, 0.0004515
3: -0.0069114, -0.0065332, -0.0069162, -0.0065332, -0.0002055, 0.0002405
4: 0.0027647, 0.0029255, 0.0027647, 0.0029275, -0.0001023, 0.0000874
5: 0.0134947, 0.0145396, 0.0134947, 0.0145530, -0.0006647, 0.0005678
6: -0.0021495, -0.0018843, -0.0021529, -0.0018843, -0.0001441, 0.0001687
7: -0.0086990, -0.0080128, -0.0087078, -0.0080128, -0.0003729, 0.0004365
8: -0.0041389, -0.0037780, -0.0041435, -0.0037780, -0.0001961, 0.0002295
9: 0.0025169, 0.0029353, 0.0025169, 0.0029407, -0.0002662, 0.0002274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002419, upper bound: 0.0002755
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002419, upper bound: 0.0002810
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9880778, 0.9887069, 0.9879955, 0.9886315, -0.0003606, 0.0005634
1: -0.0042347, -0.0040779, -0.0042552, -0.0040967, -0.0000898, 0.0001404
2: 0.0115567, 0.0123875, 0.0116563, 0.0124962, -0.0007439, 0.0004761
3: -0.0069114, -0.0065332, -0.0069608, -0.0065786, -0.0002167, 0.0003386
4: 0.0027647, 0.0029255, 0.0027839, 0.0029465, -0.0001440, 0.0000922
5: 0.0134947, 0.0145396, 0.0136200, 0.0146764, -0.0009356, 0.0005989
6: -0.0021495, -0.0018843, -0.0021842, -0.0019161, -0.0001520, 0.0002375
7: -0.0086990, -0.0080128, -0.0087888, -0.0080951, -0.0003933, 0.0006144
8: -0.0041389, -0.0037780, -0.0041861, -0.0038213, -0.0002068, 0.0003231
9: 0.0025169, 0.0029353, 0.0025671, 0.0029901, -0.0003747, 0.0002398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001792, upper bound: 0.0002301
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0001792, upper bound: 0.0002806
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9880034, 0.9886315, 0.9880697, 0.9887069, -0.0005194, 0.0004080
1: -0.0042532, -0.0040967, -0.0042367, -0.0040779, -0.0001294, 0.0001017
2: 0.0116563, 0.0124856, 0.0115567, 0.0123981, -0.0005388, 0.0006859
3: -0.0069560, -0.0065786, -0.0069162, -0.0065332, -0.0003122, 0.0002452
4: 0.0027839, 0.0029445, 0.0027647, 0.0029275, -0.0001043, 0.0001327
5: 0.0136200, 0.0146631, 0.0134947, 0.0145530, -0.0006776, 0.0008626
6: -0.0021808, -0.0019161, -0.0021529, -0.0018843, -0.0002189, 0.0001720
7: -0.0087801, -0.0080951, -0.0087078, -0.0080128, -0.0005665, 0.0004450
8: -0.0041815, -0.0038213, -0.0041435, -0.0037780, -0.0002979, 0.0002340
9: 0.0025671, 0.0029848, 0.0025169, 0.0029407, -0.0002713, 0.0003454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002004, upper bound: 0.0002246
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002004, upper bound: 0.0002738
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9880034, 0.9886315, 0.9879955, 0.9886315, -0.0003494, 0.0004040
1: -0.0042532, -0.0040967, -0.0042552, -0.0040967, -0.0000871, 0.0001007
2: 0.0116563, 0.0124856, 0.0116563, 0.0124962, -0.0005335, 0.0004613
3: -0.0069560, -0.0065786, -0.0069608, -0.0065786, -0.0002100, 0.0002428
4: 0.0027839, 0.0029445, 0.0027839, 0.0029465, -0.0001033, 0.0000893
5: 0.0136200, 0.0146631, 0.0136200, 0.0146764, -0.0006710, 0.0005802
6: -0.0021808, -0.0019161, -0.0021842, -0.0019161, -0.0001473, 0.0001703
7: -0.0087801, -0.0080951, -0.0087888, -0.0080951, -0.0003810, 0.0004406
8: -0.0041815, -0.0038213, -0.0041861, -0.0038213, -0.0002004, 0.0002317
9: 0.0025671, 0.0029848, 0.0025671, 0.0029901, -0.0002687, 0.0002324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001657, upper bound: 0.0002037
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0001657, upper bound: 0.0002738
time: 0.47 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.27 seconds
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.27
Output dim: 0, lower bound: -0.0002424, upper bound: 0.0002755
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.27
Output dim: 0, lower bound: -0.0002424, upper bound: 0.0002810
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.27
Output dim: 0, lower bound: -0.0001923, upper bound: 0.0002365
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.27
Output dim: 0, lower bound: -0.0001923, upper bound: 0.0002806
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.27
Output dim: 0, lower bound: -0.0002043, upper bound: 0.0002253
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.27
Output dim: 0, lower bound: -0.0002043, upper bound: 0.0002738
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.27
Output dim: 0, lower bound: -0.0001783, upper bound: 0.0002109
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.27
Output dim: 0, lower bound: -0.0001783, upper bound: 0.0002738
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.27
Output dim: 0, lower bound: -0.0002419, upper bound: 0.0002755
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.27
Output dim: 0, lower bound: -0.0002419, upper bound: 0.0002810
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.27
Output dim: 0, lower bound: -0.0001792, upper bound: 0.0002301
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.27
Output dim: 0, lower bound: -0.0001792, upper bound: 0.0002806
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.27
Output dim: 0, lower bound: -0.0002004, upper bound: 0.0002246
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.27
Output dim: 0, lower bound: -0.0002004, upper bound: 0.0002738
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.27
Output dim: 0, lower bound: -0.0001657, upper bound: 0.0002037
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.27
Output dim: 0, lower bound: -0.0001657, upper bound: 0.0002738

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9880778, 0.9887069, 0.9881113, 0.9886937, -0.0003955, 0.0003650
1: -0.0042347, -0.0040779, -0.0042263, -0.0040812, -0.0000986, 0.0000909
2: 0.0115567, 0.0123875, 0.0115742, 0.0123432, -0.0004820, 0.0005223
3: -0.0069114, -0.0065332, -0.0068912, -0.0065412, -0.0002377, 0.0002194
4: 0.0027647, 0.0029255, 0.0027681, 0.0029169, -0.0000933, 0.0001011
5: 0.0134947, 0.0145396, 0.0135168, 0.0144839, -0.0006062, 0.0006569
6: -0.0021495, -0.0018843, -0.0021353, -0.0018899, -0.0001667, 0.0001539
7: -0.0086990, -0.0080128, -0.0086624, -0.0080273, -0.0004314, 0.0003981
8: -0.0041389, -0.0037780, -0.0041196, -0.0037856, -0.0002268, 0.0002093
9: 0.0025169, 0.0029353, 0.0025258, 0.0029131, -0.0002427, 0.0002630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 98

Time for candidate selection: 2.14 seconds

### Candidate
type: A, layer: 3, pos: 98

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002229, upper bound: 0.0002527
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002252, upper bound: 0.0002581
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9880778, 0.9887069, 0.9880921, 0.9886588, -0.0003662, 0.0003893
1: -0.0042347, -0.0040779, -0.0042311, -0.0040899, -0.0000913, 0.0000970
2: 0.0115567, 0.0123875, 0.0116201, 0.0123685, -0.0005141, 0.0004836
3: -0.0069114, -0.0065332, -0.0069027, -0.0065621, -0.0002201, 0.0002340
4: 0.0027647, 0.0029255, 0.0027769, 0.0029218, -0.0000995, 0.0000936
5: 0.0134947, 0.0145396, 0.0135745, 0.0145158, -0.0006465, 0.0006082
6: -0.0021495, -0.0018843, -0.0021434, -0.0019045, -0.0001544, 0.0001641
7: -0.0086990, -0.0080128, -0.0086834, -0.0080652, -0.0003994, 0.0004246
8: -0.0041389, -0.0037780, -0.0041306, -0.0038056, -0.0002101, 0.0002233
9: 0.0025169, 0.0029353, 0.0025489, 0.0029258, -0.0002589, 0.0002436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 98

Time for candidate selection: 2.16 seconds

### Candidate
type: A, layer: 3, pos: 98

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002229, upper bound: 0.0002597
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002252, upper bound: 0.0002607
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9880778, 0.9887069, 0.9880054, 0.9885958, -0.0003836, 0.0005659
1: -0.0042347, -0.0040779, -0.0042527, -0.0041056, -0.0000956, 0.0001410
2: 0.0115567, 0.0123875, 0.0117034, 0.0124830, -0.0007473, 0.0005065
3: -0.0069114, -0.0065332, -0.0069548, -0.0066000, -0.0002306, 0.0003401
4: 0.0027647, 0.0029255, 0.0027931, 0.0029439, -0.0001446, 0.0000980
5: 0.0134947, 0.0145396, 0.0136793, 0.0146598, -0.0009399, 0.0006371
6: -0.0021495, -0.0018843, -0.0021800, -0.0019311, -0.0001617, 0.0002386
7: -0.0086990, -0.0080128, -0.0087779, -0.0081340, -0.0004184, 0.0006172
8: -0.0041389, -0.0037780, -0.0041803, -0.0038417, -0.0002200, 0.0003246
9: 0.0025169, 0.0029353, 0.0025908, 0.0029835, -0.0003764, 0.0002551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 98

Time for candidate selection: 2.16 seconds

### Candidate
type: A, layer: 3, pos: 98

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001708, upper bound: 0.0002591
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001763, upper bound: 0.0002603
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9880034, 0.9886315, 0.9880921, 0.9886588, -0.0004934, 0.0003684
1: -0.0042532, -0.0040967, -0.0042311, -0.0040899, -0.0001229, 0.0000918
2: 0.0116563, 0.0124856, 0.0116201, 0.0123685, -0.0004865, 0.0006516
3: -0.0069560, -0.0065786, -0.0069027, -0.0065621, -0.0002966, 0.0002214
4: 0.0027839, 0.0029445, 0.0027769, 0.0029218, -0.0000942, 0.0001261
5: 0.0136200, 0.0146631, 0.0135745, 0.0145158, -0.0006118, 0.0008195
6: -0.0021808, -0.0019161, -0.0021434, -0.0019045, -0.0002080, 0.0001553
7: -0.0087801, -0.0080951, -0.0086834, -0.0080652, -0.0005382, 0.0004018
8: -0.0041815, -0.0038213, -0.0041306, -0.0038056, -0.0002830, 0.0002113
9: 0.0025671, 0.0029848, 0.0025489, 0.0029258, -0.0002450, 0.0003282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 98
type: A, layer: 3, pos: 108

Time for candidate selection: 2.14 seconds

### Candidate
type: A, layer: 3, pos: 98

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001834, upper bound: 0.0002532
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001879, upper bound: 0.0002562
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9880034, 0.9886315, 0.9880054, 0.9885958, -0.0003800, 0.0004047
1: -0.0042532, -0.0040967, -0.0042527, -0.0041056, -0.0000947, 0.0001008
2: 0.0116563, 0.0124856, 0.0117034, 0.0124830, -0.0005343, 0.0005017
3: -0.0069560, -0.0065786, -0.0069548, -0.0066000, -0.0002284, 0.0002432
4: 0.0027839, 0.0029445, 0.0027931, 0.0029439, -0.0001034, 0.0000971
5: 0.0136200, 0.0146631, 0.0136793, 0.0146598, -0.0006721, 0.0006311
6: -0.0021808, -0.0019161, -0.0021800, -0.0019311, -0.0001602, 0.0001706
7: -0.0087801, -0.0080951, -0.0087779, -0.0081340, -0.0004144, 0.0004413
8: -0.0041815, -0.0038213, -0.0041803, -0.0038417, -0.0002179, 0.0002321
9: 0.0025671, 0.0029848, 0.0025908, 0.0029835, -0.0002691, 0.0002527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 98
type: A, layer: 3, pos: 108

Time for candidate selection: 2.27 seconds

### Candidate
type: A, layer: 3, pos: 98

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001564, upper bound: 0.0002530
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001614, upper bound: 0.0002562
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9880778, 0.9887069, 0.9880908, 0.9887484, -0.0004081, 0.0003467
1: -0.0042347, -0.0040779, -0.0042314, -0.0040675, -0.0001017, 0.0000864
2: 0.0115567, 0.0123875, 0.0115018, 0.0123703, -0.0004579, 0.0005389
3: -0.0069114, -0.0065332, -0.0069035, -0.0065083, -0.0002453, 0.0002084
4: 0.0027647, 0.0029255, 0.0027540, 0.0029221, -0.0000886, 0.0001043
5: 0.0134947, 0.0145396, 0.0134257, 0.0145180, -0.0005759, 0.0006777
6: -0.0021495, -0.0018843, -0.0021440, -0.0018668, -0.0001720, 0.0001462
7: -0.0086990, -0.0080128, -0.0086848, -0.0079675, -0.0004451, 0.0003782
8: -0.0041389, -0.0037780, -0.0041314, -0.0037542, -0.0002341, 0.0001989
9: 0.0025169, 0.0029353, 0.0024893, 0.0029267, -0.0002306, 0.0002714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 98

Time for candidate selection: 2.14 seconds

### Candidate
type: A, layer: 3, pos: 98

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002216, upper bound: 0.0002527
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002249, upper bound: 0.0002581
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9880778, 0.9887069, 0.9880778, 0.9887069, -0.0003416, 0.0003416
1: -0.0042347, -0.0040779, -0.0042347, -0.0040779, -0.0000851, 0.0000851
2: 0.0115567, 0.0123875, 0.0115567, 0.0123875, -0.0004510, 0.0004510
3: -0.0069114, -0.0065332, -0.0069114, -0.0065332, -0.0002053, 0.0002053
4: 0.0027647, 0.0029255, 0.0027647, 0.0029255, -0.0000873, 0.0000873
5: 0.0134947, 0.0145396, 0.0134947, 0.0145396, -0.0005673, 0.0005673
6: -0.0021495, -0.0018843, -0.0021495, -0.0018843, -0.0001440, 0.0001440
7: -0.0086990, -0.0080128, -0.0086990, -0.0080128, -0.0003725, 0.0003725
8: -0.0041389, -0.0037780, -0.0041389, -0.0037780, -0.0001959, 0.0001959
9: 0.0025169, 0.0029353, 0.0025169, 0.0029353, -0.0002272, 0.0002272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 98

Time for candidate selection: 2.17 seconds

### Candidate
type: A, layer: 3, pos: 98

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002216, upper bound: 0.0002597
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002249, upper bound: 0.0002607
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9880778, 0.9887069, 0.9880034, 0.9886315, -0.0003602, 0.0005191
1: -0.0042347, -0.0040779, -0.0042532, -0.0040967, -0.0000898, 0.0001293
2: 0.0115567, 0.0123875, 0.0116563, 0.0124856, -0.0006854, 0.0004757
3: -0.0069114, -0.0065332, -0.0069560, -0.0065786, -0.0002165, 0.0003120
4: 0.0027647, 0.0029255, 0.0027839, 0.0029445, -0.0001327, 0.0000921
5: 0.0134947, 0.0145396, 0.0136200, 0.0146631, -0.0008621, 0.0005983
6: -0.0021495, -0.0018843, -0.0021808, -0.0019161, -0.0001519, 0.0002188
7: -0.0086990, -0.0080128, -0.0087801, -0.0080951, -0.0003929, 0.0005661
8: -0.0041389, -0.0037780, -0.0041815, -0.0038213, -0.0002066, 0.0002977
9: 0.0025169, 0.0029353, 0.0025671, 0.0029848, -0.0003452, 0.0002396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 98

Time for candidate selection: 2.20 seconds

### Candidate
type: A, layer: 3, pos: 98

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001528, upper bound: 0.0002591
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001641, upper bound: 0.0002603
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9880034, 0.9886315, 0.9880778, 0.9887069, -0.0005191, 0.0003602
1: -0.0042532, -0.0040967, -0.0042347, -0.0040779, -0.0001293, 0.0000898
2: 0.0116563, 0.0124856, 0.0115567, 0.0123875, -0.0004757, 0.0006854
3: -0.0069560, -0.0065786, -0.0069114, -0.0065332, -0.0003120, 0.0002165
4: 0.0027839, 0.0029445, 0.0027647, 0.0029255, -0.0000921, 0.0001327
5: 0.0136200, 0.0146631, 0.0134947, 0.0145396, -0.0005983, 0.0008621
6: -0.0021808, -0.0019161, -0.0021495, -0.0018843, -0.0002188, 0.0001519
7: -0.0087801, -0.0080951, -0.0086990, -0.0080128, -0.0005661, 0.0003929
8: -0.0041815, -0.0038213, -0.0041389, -0.0037780, -0.0002977, 0.0002066
9: 0.0025671, 0.0029848, 0.0025169, 0.0029353, -0.0002396, 0.0003452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 98
type: A, layer: 3, pos: 108

Time for candidate selection: 2.23 seconds

### Candidate
type: A, layer: 3, pos: 98

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001777, upper bound: 0.0002532
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001844, upper bound: 0.0002562
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9880034, 0.9886315, 0.9880034, 0.9886315, -0.0003490, 0.0003490
1: -0.0042532, -0.0040967, -0.0042532, -0.0040967, -0.0000870, 0.0000870
2: 0.0116563, 0.0124856, 0.0116563, 0.0124856, -0.0004609, 0.0004609
3: -0.0069560, -0.0065786, -0.0069560, -0.0065786, -0.0002098, 0.0002098
4: 0.0027839, 0.0029445, 0.0027839, 0.0029445, -0.0000892, 0.0000892
5: 0.0136200, 0.0146631, 0.0136200, 0.0146631, -0.0005797, 0.0005797
6: -0.0021808, -0.0019161, -0.0021808, -0.0019161, -0.0001471, 0.0001471
7: -0.0087801, -0.0080951, -0.0087801, -0.0080951, -0.0003806, 0.0003806
8: -0.0041815, -0.0038213, -0.0041815, -0.0038213, -0.0002002, 0.0002002
9: 0.0025671, 0.0029848, 0.0025671, 0.0029848, -0.0002321, 0.0002321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 98
type: A, layer: 3, pos: 108

Time for candidate selection: 2.21 seconds

### Candidate
type: A, layer: 3, pos: 98

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001405, upper bound: 0.0002530
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001502, upper bound: 0.0002562
time: 0.50 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.50 seconds
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0002229, upper bound: 0.0002527
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0002252, upper bound: 0.0002581
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0002229, upper bound: 0.0002597
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0002252, upper bound: 0.0002607
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0001708, upper bound: 0.0002591
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0001763, upper bound: 0.0002603
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0001834, upper bound: 0.0002532
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0001879, upper bound: 0.0002562
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0001564, upper bound: 0.0002530
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0001614, upper bound: 0.0002562
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0002216, upper bound: 0.0002527
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0002249, upper bound: 0.0002581
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0002216, upper bound: 0.0002597
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0002249, upper bound: 0.0002607
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0001528, upper bound: 0.0002591
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0001641, upper bound: 0.0002603
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0001777, upper bound: 0.0002532
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0001844, upper bound: 0.0002562
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0001405, upper bound: 0.0002530
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.50
Output dim: 0, lower bound: -0.0001502, upper bound: 0.0002562

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.59 + 106.26 = 108.85 seconds
