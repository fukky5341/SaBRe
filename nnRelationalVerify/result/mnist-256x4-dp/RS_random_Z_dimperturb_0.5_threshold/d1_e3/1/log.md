## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000371


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000163, 0.0000163)
1: (-0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0006103, 0.0006103)
2: (0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0007323, 0.0007323)
3: (0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0054016, 0.0054016)
4: (-0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0004108, 0.0004108)
5: (0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0004152, 0.0004152)
6: (0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0002020, 0.0002020)
7: (-0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0013999, 0.0013999)
8: (0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0011106, 0.0011106)
9: (0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0019975, 0.0019975)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.30 + 1.63 = 2.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0005071, upper bound: 0.0005072

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004853, upper bound: 0.0004882
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004882, upper bound: 0.0004852
time: 0.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 2, lower bound: -0.0004853, upper bound: 0.0004882
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 2, lower bound: -0.0004882, upper bound: 0.0004852

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000157, 0.0000157
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005861, 0.0005881
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0007033, 0.0007057
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0051876, 0.0052051
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003959, 0.0003945
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0004001, 0.0003988
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001940, 0.0001946
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0013490, 0.0013444
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0010702, 0.0010666
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0019248, 0.0019184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004609, upper bound: 0.0004777
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004740, upper bound: 0.0004621
time: 0.78 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000157, 0.0000157
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005881, 0.0005861
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0007057, 0.0007033
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0052051, 0.0051876
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003945, 0.0003959
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003988, 0.0004001
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001946, 0.0001940
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0013444, 0.0013490
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0010666, 0.0010702
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0019184, 0.0019248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004723, upper bound: 0.0004718
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004747, upper bound: 0.0004706
time: 0.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.68 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 2, lower bound: -0.0004609, upper bound: 0.0004777
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 2, lower bound: -0.0004740, upper bound: 0.0004621
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 2, lower bound: -0.0004723, upper bound: 0.0004718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 2, lower bound: -0.0004747, upper bound: 0.0004706

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000142, 0.0000144
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005303, 0.0005409
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006364, 0.0006491
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0046937, 0.0047880
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003642, 0.0003570
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003680, 0.0003608
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001755, 0.0001790
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0012408, 0.0012164
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009844, 0.0009650
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0017706, 0.0017357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004343, upper bound: 0.0004770
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004601, upper bound: 0.0004389
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000144, 0.0000142
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005386, 0.0005323
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006464, 0.0006387
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0047675, 0.0047112
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003583, 0.0003626
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003621, 0.0003665
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001782, 0.0001761
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0012210, 0.0012355
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009686, 0.0009802
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0017422, 0.0017630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004510
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004510
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000150, 0.0000150
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005602, 0.0005612
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006722, 0.0006734
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0049582, 0.0049672
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003778, 0.0003771
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003818, 0.0003811
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001854, 0.0001857
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0012873, 0.0012850
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0010213, 0.0010194
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0018369, 0.0018335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004382, upper bound: 0.0004710
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004714, upper bound: 0.0004384
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000150, 0.0000149
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005632, 0.0005585
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006758, 0.0006702
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0049847, 0.0049435
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003760, 0.0003791
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003800, 0.0003832
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001864, 0.0001848
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0012811, 0.0012918
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0010164, 0.0010249
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0018281, 0.0018433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004625, upper bound: 0.0004596
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004596
time: 0.76 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.67 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 2, lower bound: -0.0004343, upper bound: 0.0004770
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 2, lower bound: -0.0004601, upper bound: 0.0004389
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004510
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004510
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 2, lower bound: -0.0004382, upper bound: 0.0004710
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 2, lower bound: -0.0004714, upper bound: 0.0004384
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 2, lower bound: -0.0004625, upper bound: 0.0004596
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004596

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000135, 0.0000141
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005063, 0.0005285
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006076, 0.0006342
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0044813, 0.0046775
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003558, 0.0003408
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003595, 0.0003445
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001675, 0.0001749
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0012122, 0.0011614
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009617, 0.0009214
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0017297, 0.0016572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004206, upper bound: 0.0004632
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004210, upper bound: 0.0004604
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000138, 0.0000138
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005161, 0.0005169
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006194, 0.0006203
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0045683, 0.0045756
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003480, 0.0003474
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003517, 0.0003512
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001708, 0.0001711
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011858, 0.0011839
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009408, 0.0009393
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016920, 0.0016894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004290
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004292
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000144, 0.0000142
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005384, 0.0005321
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006461, 0.0006385
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0047652, 0.0047097
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003582, 0.0003624
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003620, 0.0003663
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001782, 0.0001761
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0012206, 0.0012349
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009683, 0.0009797
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0017416, 0.0017621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004479, upper bound: 0.0004380
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004499, upper bound: 0.0004369
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000144, 0.0000142
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005384, 0.0005323
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006462, 0.0006387
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0047660, 0.0047112
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003583, 0.0003625
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003621, 0.0003663
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001782, 0.0001761
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0012210, 0.0012351
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009686, 0.0009799
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0017422, 0.0017624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004283, upper bound: 0.0004501
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004619, upper bound: 0.0004253
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000140, 0.0000143
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005250, 0.0005359
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006300, 0.0006431
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0046470, 0.0047431
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003607, 0.0003534
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003646, 0.0003572
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001737, 0.0001773
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0012292, 0.0012043
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009752, 0.0009554
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0017540, 0.0017184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002204, upper bound: 0.0002291
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002204, upper bound: 0.0002291
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000143, 0.0000140
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005355, 0.0005260
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006426, 0.0006313
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0047397, 0.0046560
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003541, 0.0003605
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003579, 0.0003643
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001772, 0.0001741
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0012066, 0.0012283
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009573, 0.0009745
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0017218, 0.0017527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002291, upper bound: 0.0002203
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002291, upper bound: 0.0002203
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000150, 0.0000149
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005629, 0.0005583
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006755, 0.0006700
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0049825, 0.0049419
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003759, 0.0003789
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003799, 0.0003830
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001863, 0.0001848
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0012807, 0.0012913
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0010161, 0.0010244
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0018275, 0.0018425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004380, upper bound: 0.0004480
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004516, upper bound: 0.0004367
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000150, 0.0000149
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005630, 0.0005585
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006756, 0.0006702
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0049831, 0.0049435
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003760, 0.0003790
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003800, 0.0003830
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001863, 0.0001848
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0012811, 0.0012914
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0010164, 0.0010246
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0018281, 0.0018428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004295, upper bound: 0.0004588
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004619, upper bound: 0.0004284
time: 0.77 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 2, lower bound: -0.0004206, upper bound: 0.0004632
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 2, lower bound: -0.0004210, upper bound: 0.0004604
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004290
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 2, lower bound: -0.0004495, upper bound: 0.0004292
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 2, lower bound: -0.0004479, upper bound: 0.0004380
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 2, lower bound: -0.0004499, upper bound: 0.0004369
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 2, lower bound: -0.0004283, upper bound: 0.0004501
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 2, lower bound: -0.0004619, upper bound: 0.0004253
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 2, lower bound: -0.0002204, upper bound: 0.0002291
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 2, lower bound: -0.0002204, upper bound: 0.0002291
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 2, lower bound: -0.0002291, upper bound: 0.0002203
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 2, lower bound: -0.0002291, upper bound: 0.0002203
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 2, lower bound: -0.0004380, upper bound: 0.0004480
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 2, lower bound: -0.0004516, upper bound: 0.0004367
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 2, lower bound: -0.0004295, upper bound: 0.0004588
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 2, lower bound: -0.0004619, upper bound: 0.0004284

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000127, 0.0000133
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004745, 0.0004993
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005694, 0.0005992
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041998, 0.0044197
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003361, 0.0003194
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003397, 0.0003228
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001570, 0.0001652
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011454, 0.0010884
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009087, 0.0008635
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016344, 0.0015531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004203, upper bound: 0.0004582
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004181, upper bound: 0.0004628
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000127, 0.0000133
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004772, 0.0004964
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005726, 0.0005957
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042235, 0.0043937
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003342, 0.0003212
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003377, 0.0003247
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001579, 0.0001643
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011387, 0.0010946
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009034, 0.0008684
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016248, 0.0015618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002046, upper bound: 0.0002153
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002046, upper bound: 0.0002153
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000138, 0.0000138
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005159, 0.0005167
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006190, 0.0006201
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0045660, 0.0045738
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003479, 0.0003473
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003516, 0.0003510
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001707, 0.0001710
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011853, 0.0011833
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009404, 0.0009388
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016914, 0.0016885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004492, upper bound: 0.0004265
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004455, upper bound: 0.0004287
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000138, 0.0000138
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005159, 0.0005169
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006191, 0.0006203
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0045666, 0.0045756
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003480, 0.0003473
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003517, 0.0003510
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001707, 0.0001711
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011858, 0.0011835
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009408, 0.0009389
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016920, 0.0016887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004359, upper bound: 0.0004163
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004367, upper bound: 0.0004154
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000136, 0.0000135
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005091, 0.0005055
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006110, 0.0006066
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0045065, 0.0044743
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003403, 0.0003427
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003439, 0.0003464
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001685, 0.0001673
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011595, 0.0011679
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009199, 0.0009266
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016546, 0.0016665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004150, upper bound: 0.0004372
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004472, upper bound: 0.0004124
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000137, 0.0000134
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005118, 0.0005025
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006141, 0.0006030
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0045297, 0.0044477
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003383, 0.0003445
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003419, 0.0003482
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001694, 0.0001663
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011527, 0.0011739
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009145, 0.0009313
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016448, 0.0016751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004496, upper bound: 0.0004347
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004435, upper bound: 0.0004366
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000137, 0.0000139
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005144, 0.0005187
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006173, 0.0006225
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0045533, 0.0045913
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003492, 0.0003463
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003529, 0.0003500
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001702, 0.0001717
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011899, 0.0011800
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009440, 0.0009362
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016979, 0.0016838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004280, upper bound: 0.0004472
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004249, upper bound: 0.0004498
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000140, 0.0000136
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005251, 0.0005083
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006301, 0.0006099
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0046476, 0.0044988
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003422, 0.0003535
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003458, 0.0003573
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001738, 0.0001682
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011659, 0.0012045
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009250, 0.0009556
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016637, 0.0017187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004616, upper bound: 0.0004233
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004558, upper bound: 0.0004250
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000135, 0.0000136
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005054, 0.0005092
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006065, 0.0006111
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0044736, 0.0045072
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003428, 0.0003402
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003465, 0.0003439
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001673, 0.0001685
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011681, 0.0011594
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009267, 0.0009198
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016668, 0.0016543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004377, upper bound: 0.0004439
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004354, upper bound: 0.0004477
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000137, 0.0000134
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005141, 0.0005008
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006169, 0.0006010
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0045503, 0.0044330
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003372, 0.0003461
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003408, 0.0003498
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001701, 0.0001657
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011489, 0.0011792
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009114, 0.0009356
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016393, 0.0016827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004163, upper bound: 0.0004359
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004508, upper bound: 0.0004110
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000141, 0.0000142
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005278, 0.0005325
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006334, 0.0006390
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0046718, 0.0047132
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003585, 0.0003553
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003623, 0.0003591
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001747, 0.0001762
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0012215, 0.0012107
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009691, 0.0009605
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0017429, 0.0017276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004292, upper bound: 0.0004542
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004276, upper bound: 0.0004585
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000144, 0.0000140
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005383, 0.0005233
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006459, 0.0006280
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0047643, 0.0046323
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003523, 0.0003624
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003561, 0.0003662
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001781, 0.0001732
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0012005, 0.0012347
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009524, 0.0009796
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0017130, 0.0017618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004372, upper bound: 0.0004150
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004511, upper bound: 0.0004114
time: 0.79 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004203, upper bound: 0.0004582
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004181, upper bound: 0.0004628
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0002046, upper bound: 0.0002153
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0002046, upper bound: 0.0002153
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004492, upper bound: 0.0004265
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004455, upper bound: 0.0004287
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004359, upper bound: 0.0004163
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004367, upper bound: 0.0004154
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004150, upper bound: 0.0004372
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004472, upper bound: 0.0004124
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004496, upper bound: 0.0004347
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004435, upper bound: 0.0004366
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004280, upper bound: 0.0004472
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004249, upper bound: 0.0004498
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004616, upper bound: 0.0004233
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004558, upper bound: 0.0004250
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004377, upper bound: 0.0004439
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004354, upper bound: 0.0004477
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004163, upper bound: 0.0004359
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004508, upper bound: 0.0004110
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004292, upper bound: 0.0004542
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004276, upper bound: 0.0004585
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004372, upper bound: 0.0004150
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 2, lower bound: -0.0004511, upper bound: 0.0004114

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000127, 0.0000134
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004764, 0.0005005
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005718, 0.0006006
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042172, 0.0044299
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003369, 0.0003207
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003405, 0.0003242
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001577, 0.0001656
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011481, 0.0010929
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009108, 0.0008671
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016382, 0.0015595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004111, upper bound: 0.0004456
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004107, upper bound: 0.0004448
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000127, 0.0000134
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004754, 0.0005013
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005705, 0.0006016
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042077, 0.0044371
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003375, 0.0003200
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003411, 0.0003234
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001573, 0.0001659
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011499, 0.0010905
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009123, 0.0008651
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016408, 0.0015560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004087, upper bound: 0.0004508
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004085, upper bound: 0.0004505
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000138, 0.0000138
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005171, 0.0005172
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006206, 0.0006207
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0045773, 0.0045781
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003482, 0.0003481
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003519, 0.0003518
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001711, 0.0001712
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011864, 0.0011862
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009413, 0.0009411
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016930, 0.0016927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004356, upper bound: 0.0004139
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004364, upper bound: 0.0004136
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000138, 0.0000138
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005161, 0.0005180
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006194, 0.0006216
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0045685, 0.0045852
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003487, 0.0003475
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003525, 0.0003512
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001708, 0.0001714
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011883, 0.0011840
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009427, 0.0009393
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016956, 0.0016894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004327, upper bound: 0.0004158
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004331, upper bound: 0.0004151
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000129, 0.0000130
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004834, 0.0004878
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005801, 0.0005854
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042789, 0.0043178
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003284, 0.0003254
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003319, 0.0003289
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001600, 0.0001614
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011190, 0.0011089
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008878, 0.0008798
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015967, 0.0015823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004356, upper bound: 0.0004139
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004327, upper bound: 0.0004160
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000130, 0.0000129
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004868, 0.0004848
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005842, 0.0005818
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043088, 0.0042912
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003264, 0.0003277
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003299, 0.0003312
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001611, 0.0001604
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011121, 0.0011167
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008823, 0.0008859
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015869, 0.0015934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004364, upper bound: 0.0004136
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004331, upper bound: 0.0004151
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000129, 0.0000131
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004826, 0.0004894
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005791, 0.0005873
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042715, 0.0043317
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003295, 0.0003249
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003330, 0.0003283
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001597, 0.0001620
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011226, 0.0011070
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008906, 0.0008782
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016019, 0.0015796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004147, upper bound: 0.0004347
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004122, upper bound: 0.0004369
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000132, 0.0000128
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004928, 0.0004789
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005914, 0.0005747
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043619, 0.0042393
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003224, 0.0003317
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003259, 0.0003353
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001631, 0.0001585
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010986, 0.0011304
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008716, 0.0008968
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015677, 0.0016130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004109
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004425, upper bound: 0.0004122
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000137, 0.0000134
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005129, 0.0005028
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006156, 0.0006034
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0045402, 0.0044507
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003385, 0.0003453
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003421, 0.0003490
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001698, 0.0001664
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011534, 0.0011766
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009151, 0.0009335
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016459, 0.0016790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004155, upper bound: 0.0004339
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004488, upper bound: 0.0004106
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000137, 0.0000134
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005119, 0.0005037
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006143, 0.0006044
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0045310, 0.0044582
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003391, 0.0003446
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003427, 0.0003483
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001694, 0.0001667
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011554, 0.0011742
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009166, 0.0009316
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016486, 0.0016756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004125, upper bound: 0.0004357
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004427, upper bound: 0.0004117
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000138, 0.0000139
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005157, 0.0005192
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006189, 0.0006231
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0045647, 0.0045957
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003495, 0.0003472
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003533, 0.0003509
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001707, 0.0001718
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011910, 0.0011830
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009449, 0.0009385
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016995, 0.0016880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004145, upper bound: 0.0004346
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004151, upper bound: 0.0004338
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000137, 0.0000139
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005147, 0.0005200
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006176, 0.0006240
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0045554, 0.0046028
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003501, 0.0003465
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003538, 0.0003502
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001703, 0.0001721
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011929, 0.0011806
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009464, 0.0009366
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0017021, 0.0016846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004122, upper bound: 0.0004369
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004124, upper bound: 0.0004358
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000141, 0.0000136
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005264, 0.0005086
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006317, 0.0006104
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0046590, 0.0045022
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003424, 0.0003543
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003461, 0.0003581
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001742, 0.0001683
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011668, 0.0012074
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009257, 0.0009579
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016649, 0.0017229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004109
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004489, upper bound: 0.0004106
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000140, 0.0000136
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005253, 0.0005096
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006303, 0.0006115
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0046493, 0.0045103
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003430, 0.0003536
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003467, 0.0003574
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001738, 0.0001686
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011689, 0.0012049
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009273, 0.0009559
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016679, 0.0017193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004431, upper bound: 0.0004124
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004434, upper bound: 0.0004119
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000135, 0.0000136
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005066, 0.0005094
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006079, 0.0006113
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0044841, 0.0045086
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003429, 0.0003410
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003466, 0.0003447
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001677, 0.0001686
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011684, 0.0011621
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009270, 0.0009219
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016673, 0.0016582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004124, upper bound: 0.0004432
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004369, upper bound: 0.0004122
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000135, 0.0000136
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005057, 0.0005104
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006068, 0.0006125
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0044758, 0.0045177
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003436, 0.0003404
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003473, 0.0003440
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001673, 0.0001689
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011708, 0.0011599
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009289, 0.0009202
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016706, 0.0016552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004109, upper bound: 0.0004469
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004346, upper bound: 0.0004145
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000130, 0.0000129
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004875, 0.0004834
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005851, 0.0005801
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043153, 0.0042789
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003254, 0.0003282
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003289, 0.0003317
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001613, 0.0001600
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011089, 0.0011183
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008798, 0.0008872
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015823, 0.0015958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004160, upper bound: 0.0004326
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004139, upper bound: 0.0004356
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000133, 0.0000127
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004990, 0.0004743
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005988, 0.0005692
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0044170, 0.0041980
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003193, 0.0003359
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003227, 0.0003395
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001651, 0.0001570
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010879, 0.0011447
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008631, 0.0009081
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015524, 0.0016334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004505, upper bound: 0.0004085
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004448, upper bound: 0.0004107
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000142, 0.0000143
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005302, 0.0005337
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006362, 0.0006405
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0046926, 0.0047241
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003593, 0.0003569
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003631, 0.0003607
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001754, 0.0001766
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0012243, 0.0012161
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009713, 0.0009648
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0017470, 0.0017353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004122, upper bound: 0.0004425
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004158, upper bound: 0.0004326
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000141, 0.0000143
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005292, 0.0005348
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006351, 0.0006418
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0046844, 0.0047341
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003601, 0.0003563
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003639, 0.0003601
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001751, 0.0001770
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0012269, 0.0012140
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009733, 0.0009631
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0017507, 0.0017323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004109, upper bound: 0.0004469
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004139, upper bound: 0.0004356
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000131, 0.0000129
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004894, 0.0004829
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005873, 0.0005795
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043317, 0.0042740
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003251, 0.0003295
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003285, 0.0003330
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001620, 0.0001598
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011076, 0.0011226
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008787, 0.0008906
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015805, 0.0016019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004369, upper bound: 0.0004122
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004347, upper bound: 0.0004147
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000133, 0.0000127
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004991, 0.0004745
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005990, 0.0005694
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0044179, 0.0041998
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003194, 0.0003360
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003228, 0.0003396
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001652, 0.0001570
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010884, 0.0011449
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008635, 0.0009083
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015531, 0.0016338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004508, upper bound: 0.0004087
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004456, upper bound: 0.0004111
time: 0.82 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004111, upper bound: 0.0004456
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004107, upper bound: 0.0004448
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004087, upper bound: 0.0004508
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004085, upper bound: 0.0004505
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004356, upper bound: 0.0004139
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004364, upper bound: 0.0004136
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004327, upper bound: 0.0004158
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004331, upper bound: 0.0004151
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004356, upper bound: 0.0004139
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004327, upper bound: 0.0004160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004364, upper bound: 0.0004136
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004331, upper bound: 0.0004151
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004147, upper bound: 0.0004347
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004122, upper bound: 0.0004369
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004109
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004425, upper bound: 0.0004122
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004155, upper bound: 0.0004339
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004488, upper bound: 0.0004106
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004125, upper bound: 0.0004357
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004427, upper bound: 0.0004117
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004145, upper bound: 0.0004346
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004151, upper bound: 0.0004338
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004122, upper bound: 0.0004369
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004124, upper bound: 0.0004358
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004109
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004489, upper bound: 0.0004106
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004431, upper bound: 0.0004124
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004434, upper bound: 0.0004119
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004124, upper bound: 0.0004432
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004369, upper bound: 0.0004122
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004109, upper bound: 0.0004469
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004346, upper bound: 0.0004145
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004160, upper bound: 0.0004326
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004139, upper bound: 0.0004356
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004505, upper bound: 0.0004085
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004448, upper bound: 0.0004107
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004122, upper bound: 0.0004425
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004158, upper bound: 0.0004326
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004109, upper bound: 0.0004469
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004139, upper bound: 0.0004356
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004369, upper bound: 0.0004122
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004347, upper bound: 0.0004147
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004508, upper bound: 0.0004087
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 2, lower bound: -0.0004456, upper bound: 0.0004111

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000127, 0.0000134
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004762, 0.0005003
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005715, 0.0006003
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042150, 0.0044280
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003368, 0.0003206
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003404, 0.0003240
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001576, 0.0001656
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011476, 0.0010923
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009104, 0.0008666
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016375, 0.0015587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 179

Time for candidate selection: 1.82 seconds

### Candidate
type: RSZ, layer: 3, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003338, upper bound: 0.0003024
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003069, upper bound: 0.0003360
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000127, 0.0000134
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004762, 0.0005005
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005715, 0.0006006
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042153, 0.0044299
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003369, 0.0003206
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003405, 0.0003240
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001576, 0.0001656
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011481, 0.0010924
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009108, 0.0008667
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016382, 0.0015588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 1.82 seconds

### Candidate
type: RSZ, layer: 3, pos: 79

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003596, upper bound: 0.0003966
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003689, upper bound: 0.0003968
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000127, 0.0000134
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004751, 0.0005011
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005702, 0.0006013
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042055, 0.0044352
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003373, 0.0003199
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003409, 0.0003233
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001572, 0.0001658
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011494, 0.0010899
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009119, 0.0008647
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016401, 0.0015552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 1.85 seconds

### Candidate
type: RSZ, layer: 3, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004052, upper bound: 0.0004371
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003953, upper bound: 0.0004473
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000127, 0.0000134
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004752, 0.0005013
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005702, 0.0006016
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042058, 0.0044371
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003375, 0.0003199
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003411, 0.0003233
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001572, 0.0001659
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011499, 0.0010900
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009123, 0.0008647
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016408, 0.0015553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 1.84 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003675, upper bound: 0.0004320
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003882, upper bound: 0.0004016
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000130, 0.0000131
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004853, 0.0004888
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005824, 0.0005865
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042957, 0.0043262
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003290, 0.0003267
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003325, 0.0003302
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001606, 0.0001617
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011212, 0.0011133
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008895, 0.0008832
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015998, 0.0015885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 179

Time for candidate selection: 1.85 seconds

### Candidate
type: RSZ, layer: 3, pos: 152

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004280, upper bound: 0.0003995
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004230, upper bound: 0.0004064
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000130, 0.0000130
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004887, 0.0004857
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005864, 0.0005828
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043254, 0.0042989
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003270, 0.0003290
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003305, 0.0003325
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001617, 0.0001607
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011141, 0.0011210
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008839, 0.0008893
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015897, 0.0015995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 1.84 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003980, upper bound: 0.0004128
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004357, upper bound: 0.0004012
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000129, 0.0000131
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004842, 0.0004896
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005810, 0.0005875
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042855, 0.0043333
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003296, 0.0003259
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003331, 0.0003294
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001602, 0.0001620
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011230, 0.0011106
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008909, 0.0008811
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016024, 0.0015848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 79

Time for candidate selection: 1.83 seconds

### Candidate
type: RSZ, layer: 3, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004147, upper bound: 0.0003958
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004095, upper bound: 0.0003976
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000130, 0.0000130
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004877, 0.0004866
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005852, 0.0005839
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043166, 0.0043067
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003275, 0.0003283
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003310, 0.0003318
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001614, 0.0001610
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011161, 0.0011187
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008855, 0.0008875
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015926, 0.0015963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 79

Time for candidate selection: 1.81 seconds

### Candidate
type: RSZ, layer: 3, pos: 152

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0003990
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004225, upper bound: 0.0004074
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000130, 0.0000131
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004854, 0.0004890
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005825, 0.0005868
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042962, 0.0043281
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003292, 0.0003268
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003327, 0.0003302
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001606, 0.0001618
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011217, 0.0011134
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008899, 0.0008833
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016005, 0.0015887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 1.84 seconds

### Candidate
type: RSZ, layer: 3, pos: 79

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003897, upper bound: 0.0003714
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003909, upper bound: 0.0003666
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000129, 0.0000131
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004842, 0.0004898
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005811, 0.0005878
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042862, 0.0043352
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003297, 0.0003260
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003332, 0.0003295
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001603, 0.0001621
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011235, 0.0011108
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008913, 0.0008813
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016031, 0.0015850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 1.87 seconds

### Candidate
type: RSZ, layer: 3, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004252, upper bound: 0.0003990
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004122, upper bound: 0.0004087
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000131, 0.0000130
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004887, 0.0004859
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005865, 0.0005831
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043260, 0.0043008
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003271, 0.0003290
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003306, 0.0003325
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001617, 0.0001608
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011146, 0.0011211
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008843, 0.0008894
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015904, 0.0015998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 1.85 seconds

### Candidate
type: RSZ, layer: 3, pos: 238

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004194, upper bound: 0.0004084
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004311, upper bound: 0.0003985
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000130, 0.0000130
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004877, 0.0004868
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005853, 0.0005842
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043172, 0.0043086
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003277, 0.0003283
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003312, 0.0003319
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001614, 0.0001611
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011166, 0.0011188
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008859, 0.0008876
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015933, 0.0015965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 79

Time for candidate selection: 1.82 seconds

### Candidate
type: RSZ, layer: 3, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003357, upper bound: 0.0003066
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003030, upper bound: 0.0003338
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000129, 0.0000131
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004845, 0.0004905
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005815, 0.0005887
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042887, 0.0043418
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003302, 0.0003262
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003337, 0.0003297
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001603, 0.0001623
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011252, 0.0011115
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008927, 0.0008818
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016056, 0.0015860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 1.83 seconds

### Candidate
type: RSZ, layer: 3, pos: 238

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004047, upper bound: 0.0004295
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004096, upper bound: 0.0004102
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000129, 0.0000131
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004835, 0.0004913
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005802, 0.0005896
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042796, 0.0043490
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003308, 0.0003255
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003343, 0.0003290
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001600, 0.0001626
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011271, 0.0011091
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008942, 0.0008799
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016083, 0.0015826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 179

Time for candidate selection: 1.84 seconds

### Candidate
type: RSZ, layer: 3, pos: 79

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003651, upper bound: 0.0003923
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003692, upper bound: 0.0003910
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000132, 0.0000128
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004947, 0.0004800
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005937, 0.0005760
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043792, 0.0042484
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003231, 0.0003331
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003266, 0.0003366
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001637, 0.0001588
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011010, 0.0011349
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008735, 0.0009004
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015710, 0.0016194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 1.83 seconds

### Candidate
type: RSZ, layer: 3, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004393, upper bound: 0.0003946
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004197, upper bound: 0.0004036
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000132, 0.0000128
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004937, 0.0004809
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005924, 0.0005771
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043695, 0.0042565
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003237, 0.0003323
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003272, 0.0003359
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001634, 0.0001591
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011031, 0.0011324
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008752, 0.0008984
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015741, 0.0016158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 1.83 seconds

### Candidate
type: RSZ, layer: 3, pos: 152

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004349, upper bound: 0.0003980
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004284, upper bound: 0.0004046
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000130, 0.0000130
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004872, 0.0004876
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005846, 0.0005851
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043120, 0.0043156
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003282, 0.0003280
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003317, 0.0003315
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001612, 0.0001614
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011184, 0.0011175
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008873, 0.0008866
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015959, 0.0015946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 1.83 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003330, upper bound: 0.0003069
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003024, upper bound: 0.0003361
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000133, 0.0000127
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004978, 0.0004770
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005974, 0.0005725
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0044062, 0.0042225
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003211, 0.0003351
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003246, 0.0003387
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001647, 0.0001579
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010943, 0.0011419
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008682, 0.0009059
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015615, 0.0016294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 223

Time for candidate selection: 1.85 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003364, upper bound: 0.0003068
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003026, upper bound: 0.0003333
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000130, 0.0000130
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004861, 0.0004884
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005834, 0.0005861
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043028, 0.0043226
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003288, 0.0003273
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003323, 0.0003307
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001609, 0.0001616
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011202, 0.0011151
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008888, 0.0008847
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015985, 0.0015912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 1.82 seconds

### Candidate
type: RSZ, layer: 3, pos: 152

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004049, upper bound: 0.0004259
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003947, upper bound: 0.0004281
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000133, 0.0000128
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004967, 0.0004779
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005961, 0.0005735
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043966, 0.0042299
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003217, 0.0003344
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003251, 0.0003380
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001644, 0.0001582
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010962, 0.0011394
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008697, 0.0009040
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015642, 0.0016259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 1.82 seconds

### Candidate
type: RSZ, layer: 3, pos: 238

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004122, upper bound: 0.0004065
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004374, upper bound: 0.0004018
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000129, 0.0000131
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004846, 0.0004907
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005816, 0.0005889
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042895, 0.0043437
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003304, 0.0003262
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003339, 0.0003297
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001604, 0.0001624
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011257, 0.0011117
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008931, 0.0008819
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016063, 0.0015862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 1.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003962, upper bound: 0.0004108
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003953, upper bound: 0.0004166
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000130, 0.0000130
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004872, 0.0004878
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005847, 0.0005854
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043128, 0.0043175
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003284, 0.0003280
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003319, 0.0003315
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001612, 0.0001614
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011189, 0.0011177
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008877, 0.0008867
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015966, 0.0015949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 1.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003330, upper bound: 0.0003069
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003024, upper bound: 0.0003361
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000129, 0.0000131
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004836, 0.0004916
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005803, 0.0005899
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042803, 0.0043509
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003309, 0.0003255
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003344, 0.0003290
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001600, 0.0001627
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011276, 0.0011093
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008946, 0.0008801
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016090, 0.0015829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195

Time for candidate selection: 1.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 79

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003651, upper bound: 0.0003923
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003692, upper bound: 0.0003910
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000130, 0.0000130
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004862, 0.0004886
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005835, 0.0005863
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043035, 0.0043245
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003289, 0.0003273
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003324, 0.0003308
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001609, 0.0001617
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011207, 0.0011153
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008891, 0.0008848
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015992, 0.0015914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 1.84 seconds

### Candidate
type: RSZ, layer: 3, pos: 152

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004047, upper bound: 0.0004247
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003947, upper bound: 0.0004281
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000132, 0.0000128
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004948, 0.0004802
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005938, 0.0005762
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043801, 0.0042503
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003233, 0.0003331
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003267, 0.0003367
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001638, 0.0001589
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011015, 0.0011351
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008739, 0.0009006
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015717, 0.0016197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 1.87 seconds

### Candidate
type: RSZ, layer: 3, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004395, upper bound: 0.0003947
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004198, upper bound: 0.0004036
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000133, 0.0000127
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004979, 0.0004773
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005975, 0.0005727
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0044071, 0.0042244
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003213, 0.0003352
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003247, 0.0003388
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001648, 0.0001579
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010948, 0.0011421
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008686, 0.0009061
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015622, 0.0016297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 1.85 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003364, upper bound: 0.0003068
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003026, upper bound: 0.0003333
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000132, 0.0000128
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004938, 0.0004811
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005925, 0.0005773
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043704, 0.0042584
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003239, 0.0003324
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003273, 0.0003359
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001634, 0.0001592
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011036, 0.0011326
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008755, 0.0008986
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015748, 0.0016162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 1.84 seconds

### Candidate
type: RSZ, layer: 3, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004397, upper bound: 0.0004062
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004183, upper bound: 0.0004089
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000133, 0.0000128
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004968, 0.0004781
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005962, 0.0005737
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043974, 0.0042318
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003219, 0.0003345
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003253, 0.0003380
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001644, 0.0001582
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010967, 0.0011396
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008701, 0.0009041
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015649, 0.0016262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 1.82 seconds

### Candidate
type: RSZ, layer: 3, pos: 238

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004120, upper bound: 0.0004067
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004381, upper bound: 0.0004031
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000128, 0.0000132
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004808, 0.0004938
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005770, 0.0005925
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042558, 0.0043704
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003324, 0.0003237
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003359, 0.0003271
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001591, 0.0001634
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011326, 0.0011029
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008986, 0.0008750
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016162, 0.0015738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 152

Time for candidate selection: 1.84 seconds

### Candidate
type: RSZ, layer: 3, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004089, upper bound: 0.0004183
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004062, upper bound: 0.0004398
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000131, 0.0000129
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004912, 0.0004836
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005895, 0.0005803
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043481, 0.0042803
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003255, 0.0003307
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003290, 0.0003342
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001626, 0.0001600
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011093, 0.0011268
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008801, 0.0008940
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015829, 0.0016079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 1.84 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003921, upper bound: 0.0003920
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004183, upper bound: 0.0003709
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000128, 0.0000132
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004799, 0.0004948
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005759, 0.0005938
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042476, 0.0043801
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003331, 0.0003231
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003367, 0.0003265
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001588, 0.0001638
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011351, 0.0011008
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009006, 0.0008733
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016197, 0.0015708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 179

Time for candidate selection: 1.82 seconds

### Candidate
type: RSZ, layer: 3, pos: 238

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003974, upper bound: 0.0004417
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004057, upper bound: 0.0004258
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000131, 0.0000129
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004904, 0.0004846
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005885, 0.0005816
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043409, 0.0042895
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003262, 0.0003302
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003297, 0.0003337
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001623, 0.0001604
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011117, 0.0011250
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008819, 0.0008925
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015862, 0.0016053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 79

Time for candidate selection: 1.85 seconds

### Candidate
type: RSZ, layer: 3, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003946, upper bound: 0.0004137
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004339, upper bound: 0.0004001
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000131, 0.0000129
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004895, 0.0004842
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005874, 0.0005811
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043325, 0.0042862
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003260, 0.0003295
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003295, 0.0003330
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001620, 0.0001603
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011108, 0.0011228
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008813, 0.0008908
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015850, 0.0016022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 1.84 seconds

### Candidate
type: RSZ, layer: 3, pos: 238

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004053, upper bound: 0.0004275
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004108, upper bound: 0.0004093
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000130, 0.0000130
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004887, 0.0004854
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005864, 0.0005825
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043253, 0.0042962
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003268, 0.0003290
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003302, 0.0003325
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001617, 0.0001606
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011134, 0.0011210
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008833, 0.0008893
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015887, 0.0015995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 195

Time for candidate selection: 1.86 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003957, upper bound: 0.0004130
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003934, upper bound: 0.0004177
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000134, 0.0000127
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005010, 0.0004752
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006012, 0.0005702
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0044342, 0.0042058
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003199, 0.0003372
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003233, 0.0003408
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001658, 0.0001572
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010900, 0.0011492
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008647, 0.0009117
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015553, 0.0016398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 1.84 seconds

### Candidate
type: RSZ, layer: 3, pos: 152

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004429, upper bound: 0.0003927
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004336, upper bound: 0.0004010
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000134, 0.0000127
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005002, 0.0004762
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006002, 0.0005715
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0044271, 0.0042153
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003206, 0.0003367
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003240, 0.0003403
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001655, 0.0001576
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010924, 0.0011473
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008667, 0.0009102
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015588, 0.0016371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 238

Time for candidate selection: 1.83 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003360, upper bound: 0.0003069
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003024, upper bound: 0.0003338
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000128, 0.0000132
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004809, 0.0004940
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005771, 0.0005928
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042565, 0.0043723
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003325, 0.0003237
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003361, 0.0003272
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001591, 0.0001635
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011331, 0.0011031
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008990, 0.0008752
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016169, 0.0015741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 1.86 seconds

### Candidate
type: RSZ, layer: 3, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004047, upper bound: 0.0004184
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003949, upper bound: 0.0004351
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000131, 0.0000129
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004896, 0.0004845
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005875, 0.0005814
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043333, 0.0042881
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003261, 0.0003296
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003296, 0.0003331
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001620, 0.0001603
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011113, 0.0011230
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008817, 0.0008909
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015857, 0.0016024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 223

Time for candidate selection: 1.89 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003330, upper bound: 0.0003069
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003024, upper bound: 0.0003361
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000128, 0.0000132
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004800, 0.0004951
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005760, 0.0005941
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042484, 0.0043820
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003333, 0.0003231
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003368, 0.0003266
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001588, 0.0001638
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011356, 0.0011010
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009009, 0.0008735
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016204, 0.0015710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 152

Time for candidate selection: 1.84 seconds

### Candidate
type: RSZ, layer: 3, pos: 238

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003968, upper bound: 0.0004416
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004057, upper bound: 0.0004258
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000131, 0.0000130
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004888, 0.0004856
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005865, 0.0005827
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043262, 0.0042981
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003269, 0.0003290
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003304, 0.0003325
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001617, 0.0001607
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011139, 0.0011212
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008837, 0.0008895
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015894, 0.0015998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 152

Time for candidate selection: 1.85 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004016, upper bound: 0.0004349
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004131, upper bound: 0.0003980
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000131, 0.0000129
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004913, 0.0004838
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005896, 0.0005806
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043490, 0.0042822
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003257, 0.0003308
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003292, 0.0003343
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001626, 0.0001601
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011098, 0.0011271
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008804, 0.0008942
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015836, 0.0016083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 195

Time for candidate selection: 1.86 seconds

### Candidate
type: RSZ, layer: 3, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 79

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003910, upper bound: 0.0003692
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003922, upper bound: 0.0003651
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000131, 0.0000129
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004905, 0.0004848
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005887, 0.0005818
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043418, 0.0042914
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003264, 0.0003302
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003299, 0.0003337
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001623, 0.0001604
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011121, 0.0011252
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008823, 0.0008927
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015869, 0.0016056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 1.84 seconds

### Candidate
type: RSZ, layer: 3, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004311, upper bound: 0.0003994
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004274, upper bound: 0.0004112
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000134, 0.0000127
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005011, 0.0004754
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006013, 0.0005705
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0044352, 0.0042077
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003200, 0.0003373
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003234, 0.0003409
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001658, 0.0001573
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010905, 0.0011494
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008651, 0.0009119
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015560, 0.0016401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 1.86 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004328, upper bound: 0.0003886
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004278, upper bound: 0.0003904
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000134, 0.0000127
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0005003, 0.0004764
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0006003, 0.0005718
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0044280, 0.0042172
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003207, 0.0003368
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003242, 0.0003404
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001656, 0.0001577
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010929, 0.0011476
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008671, 0.0009104
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015595, 0.0016375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 1.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 79

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003970, upper bound: 0.0003696
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003967, upper bound: 0.0003603
time: 1.01 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003338, upper bound: 0.0003024
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003069, upper bound: 0.0003360
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003596, upper bound: 0.0003966
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003689, upper bound: 0.0003968
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004052, upper bound: 0.0004371
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003953, upper bound: 0.0004473
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003675, upper bound: 0.0004320
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003882, upper bound: 0.0004016
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004280, upper bound: 0.0003995
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004230, upper bound: 0.0004064
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003980, upper bound: 0.0004128
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004357, upper bound: 0.0004012
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004147, upper bound: 0.0003958
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004095, upper bound: 0.0003976
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004254, upper bound: 0.0003990
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004225, upper bound: 0.0004074
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003897, upper bound: 0.0003714
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003909, upper bound: 0.0003666
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004252, upper bound: 0.0003990
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004122, upper bound: 0.0004087
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004194, upper bound: 0.0004084
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004311, upper bound: 0.0003985
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003357, upper bound: 0.0003066
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003030, upper bound: 0.0003338
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004047, upper bound: 0.0004295
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004096, upper bound: 0.0004102
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003651, upper bound: 0.0003923
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003692, upper bound: 0.0003910
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004393, upper bound: 0.0003946
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004197, upper bound: 0.0004036
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004349, upper bound: 0.0003980
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004284, upper bound: 0.0004046
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003330, upper bound: 0.0003069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003024, upper bound: 0.0003361
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003364, upper bound: 0.0003068
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003026, upper bound: 0.0003333
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004049, upper bound: 0.0004259
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003947, upper bound: 0.0004281
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004122, upper bound: 0.0004065
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004374, upper bound: 0.0004018
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003962, upper bound: 0.0004108
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003953, upper bound: 0.0004166
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003330, upper bound: 0.0003069
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003024, upper bound: 0.0003361
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003651, upper bound: 0.0003923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003692, upper bound: 0.0003910
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004047, upper bound: 0.0004247
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003947, upper bound: 0.0004281
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004395, upper bound: 0.0003947
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004198, upper bound: 0.0004036
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003364, upper bound: 0.0003068
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003026, upper bound: 0.0003333
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004397, upper bound: 0.0004062
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004183, upper bound: 0.0004089
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004120, upper bound: 0.0004067
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004381, upper bound: 0.0004031
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004089, upper bound: 0.0004183
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004062, upper bound: 0.0004398
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003921, upper bound: 0.0003920
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004183, upper bound: 0.0003709
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003974, upper bound: 0.0004417
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004057, upper bound: 0.0004258
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003946, upper bound: 0.0004137
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004339, upper bound: 0.0004001
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004053, upper bound: 0.0004275
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004108, upper bound: 0.0004093
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003957, upper bound: 0.0004130
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003934, upper bound: 0.0004177
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004429, upper bound: 0.0003927
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004336, upper bound: 0.0004010
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003360, upper bound: 0.0003069
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003024, upper bound: 0.0003338
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004047, upper bound: 0.0004184
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003949, upper bound: 0.0004351
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003330, upper bound: 0.0003069
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003024, upper bound: 0.0003361
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003968, upper bound: 0.0004416
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004057, upper bound: 0.0004258
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004016, upper bound: 0.0004349
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004131, upper bound: 0.0003980
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003910, upper bound: 0.0003692
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003922, upper bound: 0.0003651
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004311, upper bound: 0.0003994
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004274, upper bound: 0.0004112
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004328, upper bound: 0.0003886
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0004278, upper bound: 0.0003904
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003970, upper bound: 0.0003696
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 2, lower bound: -0.0003967, upper bound: 0.0003603

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000116, 0.0000121
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004342, 0.0004519
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005211, 0.0005423
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0038436, 0.0040002
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003042, 0.0002923
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003075, 0.0002955
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001437, 0.0001496
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010367, 0.0009961
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008225, 0.0007903
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014793, 0.0014214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 152

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003519, upper bound: 0.0003840
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003387, upper bound: 0.0003889
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000117, 0.0000122
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004371, 0.0004585
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005246, 0.0005502
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0038691, 0.0040583
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003087, 0.0002943
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003120, 0.0002974
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001447, 0.0001517
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010517, 0.0010027
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008344, 0.0007955
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015008, 0.0014308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003614, upper bound: 0.0003745
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003509, upper bound: 0.0003894
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000125, 0.0000131
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004667, 0.0004918
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005600, 0.0005902
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041305, 0.0043535
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003311, 0.0003142
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003346, 0.0003175
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001544, 0.0001628
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011283, 0.0010705
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008951, 0.0008493
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016099, 0.0015275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003300, upper bound: 0.0002779
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003037, upper bound: 0.0003193
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000124, 0.0000132
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004659, 0.0004929
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005591, 0.0005916
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041238, 0.0043632
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003319, 0.0003136
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003354, 0.0003170
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001542, 0.0001631
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011308, 0.0010687
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008971, 0.0008479
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016135, 0.0015250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003566, upper bound: 0.0004290
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003749, upper bound: 0.0003980
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000123, 0.0000133
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004603, 0.0004971
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005524, 0.0005965
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0040744, 0.0043998
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003346, 0.0003099
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003382, 0.0003132
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001523, 0.0001645
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011403, 0.0010559
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0009046, 0.0008377
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0016271, 0.0015067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 152

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003599, upper bound: 0.0004161
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003521, upper bound: 0.0004245
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000125, 0.0000130
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004672, 0.0004864
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005607, 0.0005838
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041353, 0.0043057
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003275, 0.0003145
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003310, 0.0003179
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001546, 0.0001610
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011159, 0.0010717
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008853, 0.0008502
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015922, 0.0015292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002212, upper bound: 0.0002207
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002211, upper bound: 0.0002211
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000122, 0.0000122
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004565, 0.0004582
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005478, 0.0005498
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0040405, 0.0040556
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003084, 0.0003073
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003117, 0.0003106
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001511, 0.0001516
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010510, 0.0010471
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008338, 0.0008307
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014997, 0.0014942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 238

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004108, upper bound: 0.0003942
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004226, upper bound: 0.0003834
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000121, 0.0000123
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004547, 0.0004619
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005457, 0.0005543
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0040250, 0.0040884
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003109, 0.0003061
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003143, 0.0003094
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001505, 0.0001529
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010595, 0.0010431
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008406, 0.0008276
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015119, 0.0014885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004159, upper bound: 0.0003906
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004062, upper bound: 0.0003994
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000131, 0.0000130
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004889, 0.0004867
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005867, 0.0005841
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043276, 0.0043081
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003277, 0.0003291
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003312, 0.0003327
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001618, 0.0001611
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011165, 0.0011215
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008858, 0.0008898
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015931, 0.0016003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003726, upper bound: 0.0003923
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003750, upper bound: 0.0003715
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000131, 0.0000130
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004904, 0.0004859
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005885, 0.0005831
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043405, 0.0043011
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003271, 0.0003301
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003306, 0.0003336
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001623, 0.0001608
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011147, 0.0011249
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008843, 0.0008924
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015906, 0.0016051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004178, upper bound: 0.0003825
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004125, upper bound: 0.0003826
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000127, 0.0000127
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004749, 0.0004766
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005699, 0.0005719
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042038, 0.0042186
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003208, 0.0003197
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003243, 0.0003231
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001572, 0.0001577
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010933, 0.0010895
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008674, 0.0008643
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015600, 0.0015546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 238

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003906, upper bound: 0.0003905
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004094, upper bound: 0.0003855
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000126, 0.0000128
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004712, 0.0004782
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005655, 0.0005738
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041708, 0.0042326
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003219, 0.0003172
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003253, 0.0003206
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001559, 0.0001582
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010969, 0.0010809
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008702, 0.0008575
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015652, 0.0015424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004017, upper bound: 0.0003802
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003861, upper bound: 0.0003898
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000123, 0.0000122
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004591, 0.0004560
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005509, 0.0005472
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0040636, 0.0040361
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003070, 0.0003091
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003102, 0.0003124
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001519, 0.0001509
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010460, 0.0010531
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008298, 0.0008355
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014925, 0.0015027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004218, upper bound: 0.0003922
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004112, upper bound: 0.0003955
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000122, 0.0000123
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004571, 0.0004596
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005485, 0.0005515
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0040460, 0.0040680
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003094, 0.0003077
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003127, 0.0003110
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001513, 0.0001521
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010542, 0.0010485
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008364, 0.0008319
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015043, 0.0014962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003218, upper bound: 0.0002991
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002885, upper bound: 0.0003263
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000118, 0.0000119
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004434, 0.0004457
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005321, 0.0005349
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0039246, 0.0039452
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003001, 0.0002985
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003033, 0.0003017
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001467, 0.0001475
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010224, 0.0010171
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008112, 0.0008069
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014589, 0.0014513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002909, upper bound: 0.0002613
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002578, upper bound: 0.0002918
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000118, 0.0000119
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004422, 0.0004470
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005307, 0.0005364
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0039143, 0.0039565
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003009, 0.0002977
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003041, 0.0003009
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001464, 0.0001479
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010254, 0.0010144
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008135, 0.0008048
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014631, 0.0014475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002950, upper bound: 0.0002529
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002665, upper bound: 0.0002837
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000122, 0.0000122
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004551, 0.0004566
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005462, 0.0005480
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0040286, 0.0040418
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003074, 0.0003064
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003107, 0.0003097
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001506, 0.0001511
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010475, 0.0010440
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008310, 0.0008283
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014946, 0.0014898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 152

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004179, upper bound: 0.0003900
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004156, upper bound: 0.0003916
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000120, 0.0000123
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004511, 0.0004608
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005413, 0.0005530
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0039926, 0.0040791
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003102, 0.0003037
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003135, 0.0003069
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001493, 0.0001525
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010571, 0.0010347
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008387, 0.0008209
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015084, 0.0014764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 152

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003164, upper bound: 0.0002957
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002881, upper bound: 0.0003264
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000125, 0.0000126
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004690, 0.0004730
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005628, 0.0005676
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041509, 0.0041869
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003184, 0.0003157
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003218, 0.0003191
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001552, 0.0001565
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010851, 0.0010757
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008608, 0.0008534
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015483, 0.0015350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004159, upper bound: 0.0003988
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004072, upper bound: 0.0004049
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000127, 0.0000125
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004759, 0.0004688
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005711, 0.0005625
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042121, 0.0041491
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003156, 0.0003204
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003189, 0.0003238
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001575, 0.0001551
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010753, 0.0010916
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008531, 0.0008660
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015343, 0.0015576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003918, upper bound: 0.0003977
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004304, upper bound: 0.0003910
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000124, 0.0000128
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004644, 0.0004777
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005573, 0.0005732
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041107, 0.0042279
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003216, 0.0003126
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003250, 0.0003160
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001537, 0.0001581
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010957, 0.0010653
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008693, 0.0008452
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015635, 0.0015201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 79

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003568, upper bound: 0.0003848
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003618, upper bound: 0.0003835
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000126, 0.0000126
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004717, 0.0004718
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005660, 0.0005662
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041748, 0.0041763
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003176, 0.0003175
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003210, 0.0003209
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001561, 0.0001561
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010823, 0.0010819
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008587, 0.0008584
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015444, 0.0015438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 152

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004020, upper bound: 0.0003973
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003893, upper bound: 0.0004024
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000118, 0.0000119
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004415, 0.0004470
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005298, 0.0005365
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0039080, 0.0039569
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003009, 0.0002972
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003042, 0.0003004
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001461, 0.0001479
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010255, 0.0010128
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008136, 0.0008035
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014633, 0.0014452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003203, upper bound: 0.0003731
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003449, upper bound: 0.0003492
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000117, 0.0000120
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004386, 0.0004494
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005264, 0.0005392
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0038825, 0.0039774
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003025, 0.0002953
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003057, 0.0002984
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001452, 0.0001487
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010308, 0.0010062
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008178, 0.0007983
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014708, 0.0014358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003509, upper bound: 0.0003677
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003495, upper bound: 0.0003731
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000125, 0.0000119
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004667, 0.0004468
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005600, 0.0005362
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041308, 0.0039547
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003008, 0.0003142
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003040, 0.0003175
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001544, 0.0001479
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010249, 0.0010705
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008131, 0.0008493
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014624, 0.0015275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004146, upper bound: 0.0003939
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004386, upper bound: 0.0003680
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000123, 0.0000120
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004616, 0.0004495
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005539, 0.0005394
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0040855, 0.0039785
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003026, 0.0003107
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003058, 0.0003140
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001528, 0.0001487
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010311, 0.0010588
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008180, 0.0008400
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014712, 0.0015108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 79

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004016, upper bound: 0.0003827
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003958, upper bound: 0.0003849
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000125, 0.0000120
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004672, 0.0004503
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005607, 0.0005404
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041356, 0.0039859
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003031, 0.0003145
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003064, 0.0003179
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001546, 0.0001490
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010330, 0.0010718
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008195, 0.0008503
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014740, 0.0015293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 79

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003872, upper bound: 0.0003574
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003870, upper bound: 0.0003449
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000124, 0.0000121
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004631, 0.0004517
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005557, 0.0005420
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0040989, 0.0039980
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003041, 0.0003117
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003073, 0.0003151
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001532, 0.0001495
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010361, 0.0010623
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008220, 0.0008427
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014785, 0.0015158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 238

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003951, upper bound: 0.0003992
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004230, upper bound: 0.0003942
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000123, 0.0000122
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004595, 0.0004578
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005514, 0.0005494
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0040673, 0.0040520
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003082, 0.0003093
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003115, 0.0003126
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001521, 0.0001515
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010501, 0.0010541
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008331, 0.0008362
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014984, 0.0015041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003251, upper bound: 0.0002923
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002947, upper bound: 0.0003229
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000122, 0.0000123
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004555, 0.0004588
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005467, 0.0005505
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0040321, 0.0040606
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003088, 0.0003067
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003121, 0.0003099
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001508, 0.0001518
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010523, 0.0010450
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008349, 0.0008290
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015016, 0.0014911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003544, upper bound: 0.0004097
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003744, upper bound: 0.0003817
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000127, 0.0000124
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004758, 0.0004650
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005710, 0.0005580
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042117, 0.0041160
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003130, 0.0003203
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003164, 0.0003237
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001575, 0.0001539
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010667, 0.0010915
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008463, 0.0008659
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015221, 0.0015575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 152

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 79

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003664, upper bound: 0.0003648
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003677, upper bound: 0.0003553
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000129, 0.0000123
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004838, 0.0004606
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005806, 0.0005527
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042827, 0.0040767
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003101, 0.0003257
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003134, 0.0003292
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001601, 0.0001524
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010565, 0.0011099
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008382, 0.0008805
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015076, 0.0015837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004102, upper bound: 0.0004010
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004366, upper bound: 0.0003862
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000126, 0.0000128
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004732, 0.0004778
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005678, 0.0005734
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041881, 0.0042293
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003217, 0.0003185
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003251, 0.0003219
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001566, 0.0001581
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010961, 0.0010854
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008696, 0.0008611
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015640, 0.0015487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 152

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003882, upper bound: 0.0003974
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003762, upper bound: 0.0004027
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000126, 0.0000129
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004717, 0.0004819
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005660, 0.0005783
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041748, 0.0042658
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003244, 0.0003175
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003279, 0.0003209
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001561, 0.0001595
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011055, 0.0010819
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008771, 0.0008584
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015775, 0.0015438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003874, upper bound: 0.0003965
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003757, upper bound: 0.0004087
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000118, 0.0000120
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004416, 0.0004488
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005299, 0.0005386
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0039087, 0.0039727
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003021, 0.0002973
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003054, 0.0003005
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001461, 0.0001485
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010296, 0.0010130
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008168, 0.0008036
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014691, 0.0014454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002841, upper bound: 0.0002624
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002556, upper bound: 0.0002946
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000117, 0.0000120
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004388, 0.0004496
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005266, 0.0005395
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0038841, 0.0039793
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003026, 0.0002954
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003059, 0.0002986
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001452, 0.0001488
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010313, 0.0010066
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008182, 0.0007986
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014715, 0.0014363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003509, upper bound: 0.0003677
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003495, upper bound: 0.0003731
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000123, 0.0000122
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004600, 0.0004580
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005521, 0.0005496
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0040718, 0.0040537
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003083, 0.0003097
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003116, 0.0003130
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001522, 0.0001516
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010506, 0.0010553
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008335, 0.0008372
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014991, 0.0015058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004012, upper bound: 0.0004170
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003895, upper bound: 0.0004212
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000122, 0.0000123
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004556, 0.0004600
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005468, 0.0005520
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0040328, 0.0040714
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003097, 0.0003067
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003130, 0.0003100
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001508, 0.0001522
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010552, 0.0010451
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008371, 0.0008292
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015056, 0.0014913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003171, upper bound: 0.0002996
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002866, upper bound: 0.0003289
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000125, 0.0000119
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004670, 0.0004470
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005604, 0.0005365
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041335, 0.0039569
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003009, 0.0003144
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003042, 0.0003177
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001545, 0.0001479
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010255, 0.0010712
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008136, 0.0008499
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014633, 0.0015286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004359, upper bound: 0.0003905
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004161, upper bound: 0.0003910
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000123, 0.0000120
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004617, 0.0004503
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005540, 0.0005404
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0040864, 0.0039860
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003032, 0.0003108
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003064, 0.0003141
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001528, 0.0001490
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010330, 0.0010590
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008195, 0.0008402
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014740, 0.0015111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 79

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003737, upper bound: 0.0003829
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004017, upper bound: 0.0003653
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000130, 0.0000126
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004856, 0.0004719
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005828, 0.0005663
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042986, 0.0041768
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003177, 0.0003269
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003211, 0.0003304
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001607, 0.0001562
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010824, 0.0011140
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008588, 0.0008838
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015446, 0.0015896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 79

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 238

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004094, upper bound: 0.0004010
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004345, upper bound: 0.0003943
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000129, 0.0000126
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004845, 0.0004725
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005815, 0.0005670
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042887, 0.0041819
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003181, 0.0003262
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003215, 0.0003297
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001603, 0.0001564
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010838, 0.0011115
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008598, 0.0008818
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015464, 0.0015860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003908, upper bound: 0.0004081
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004176, upper bound: 0.0003896
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000127, 0.0000124
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004765, 0.0004652
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005719, 0.0005583
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042180, 0.0041179
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003132, 0.0003208
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003165, 0.0003242
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001577, 0.0001540
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010672, 0.0010931
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008466, 0.0008672
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015228, 0.0015598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003975, upper bound: 0.0004060
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004112, upper bound: 0.0003880
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000129, 0.0000123
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004839, 0.0004620
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005808, 0.0005545
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042835, 0.0040897
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003110, 0.0003258
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003144, 0.0003293
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001602, 0.0001529
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010599, 0.0011101
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008408, 0.0008807
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015123, 0.0015840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004199, upper bound: 0.0003842
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004161, upper bound: 0.0003844
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000126, 0.0000129
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004723, 0.0004845
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005667, 0.0005815
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041800, 0.0042887
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003262, 0.0003179
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003297, 0.0003213
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001563, 0.0001603
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011115, 0.0010833
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008818, 0.0008594
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015860, 0.0015458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003906, upper bound: 0.0003965
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003886, upper bound: 0.0004001
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000126, 0.0000130
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004716, 0.0004856
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005659, 0.0005828
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041742, 0.0042986
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003269, 0.0003175
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003304, 0.0003209
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001561, 0.0001607
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011140, 0.0010818
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008838, 0.0008582
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015896, 0.0015436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003876, upper bound: 0.0004390
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004054, upper bound: 0.0004120
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000127, 0.0000127
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004764, 0.0004746
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005717, 0.0005696
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042167, 0.0042010
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003195, 0.0003207
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003229, 0.0003241
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001577, 0.0001571
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010887, 0.0010928
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008637, 0.0008670
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015535, 0.0015593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003850, upper bound: 0.0003748
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003713, upper bound: 0.0003844
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000130, 0.0000125
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004875, 0.0004687
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005850, 0.0005625
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043150, 0.0041489
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003155, 0.0003282
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003189, 0.0003317
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001613, 0.0001551
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010752, 0.0011183
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008530, 0.0008872
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015343, 0.0015957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002211, upper bound: 0.0002211
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002208, upper bound: 0.0002212
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000123, 0.0000129
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004622, 0.0004820
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005546, 0.0005784
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0040907, 0.0042661
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003245, 0.0003111
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003279, 0.0003144
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001529, 0.0001595
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011056, 0.0010601
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008771, 0.0008411
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015776, 0.0015127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003940, upper bound: 0.0004185
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003855, upper bound: 0.0004382
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000125, 0.0000127
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004670, 0.0004741
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005604, 0.0005690
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041337, 0.0041968
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003192, 0.0003144
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003226, 0.0003177
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001546, 0.0001569
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010876, 0.0010713
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008629, 0.0008499
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015520, 0.0015286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004021, upper bound: 0.0004015
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003995, upper bound: 0.0004223
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000131, 0.0000130
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004907, 0.0004857
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005888, 0.0005829
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043431, 0.0042992
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003270, 0.0003303
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003305, 0.0003338
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001624, 0.0001607
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011142, 0.0011256
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008839, 0.0008930
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015898, 0.0016061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 238

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003873, upper bound: 0.0004085
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003895, upper bound: 0.0004026
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000131, 0.0000129
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004921, 0.0004849
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005906, 0.0005819
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0043559, 0.0042916
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003264, 0.0003313
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003299, 0.0003348
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001629, 0.0001605
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0011122, 0.0011289
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008824, 0.0008956
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015870, 0.0016108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003749, upper bound: 0.0003774
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004160, upper bound: 0.0003744
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000125, 0.0000126
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004697, 0.0004714
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005636, 0.0005657
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041572, 0.0041723
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003173, 0.0003162
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003207, 0.0003196
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001554, 0.0001560
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010813, 0.0010774
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008578, 0.0008547
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015429, 0.0015373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003979, upper bound: 0.0004068
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003883, upper bound: 0.0004198
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000127, 0.0000124
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004766, 0.0004662
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005720, 0.0005594
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042186, 0.0041261
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003138, 0.0003208
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003172, 0.0003243
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001577, 0.0001543
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010693, 0.0010933
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008483, 0.0008674
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015258, 0.0015600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 79

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003642, upper bound: 0.0003664
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003676, upper bound: 0.0003617
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000127, 0.0000126
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004769, 0.0004724
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005723, 0.0005669
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042211, 0.0041815
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003180, 0.0003210
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003214, 0.0003245
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001578, 0.0001563
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010837, 0.0010939
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008597, 0.0008679
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015463, 0.0015610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 79

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003878, upper bound: 0.0003881
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003796, upper bound: 0.0004053
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000127, 0.0000127
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004757, 0.0004764
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005709, 0.0005716
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0042107, 0.0042164
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003207, 0.0003202
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003241, 0.0003237
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001574, 0.0001576
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010927, 0.0010912
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008669, 0.0008657
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0015592, 0.0015571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 152
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 79

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003463, upper bound: 0.0003729
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0003508, upper bound: 0.0003718
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000126, 0.0000119
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004726, 0.0004446
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005671, 0.0005335
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041832, 0.0039352
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0002993, 0.0003182
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003025, 0.0003216
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001564, 0.0001471
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010198, 0.0010841
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008091, 0.0008601
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014552, 0.0015469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003290, upper bound: 0.0002919
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0002952, upper bound: 0.0003180
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041067, -0.0040765, -0.0041067, -0.0040765, -0.0000126, 0.0000120
1: -0.0064536, -0.0053209, -0.0064536, -0.0053209, -0.0004704, 0.0004476
2: 0.9687189, 0.9700781, 0.9687189, 0.9700781, -0.0005645, 0.0005371
3: 0.0155814, 0.0256073, 0.0155814, 0.0256073, -0.0041636, 0.0039616
4: -0.0026406, -0.0018781, -0.0026406, -0.0018781, -0.0003013, 0.0003167
5: 0.0146015, 0.0153722, 0.0146015, 0.0153722, -0.0003045, 0.0003200
6: 0.0044297, 0.0048046, 0.0044297, 0.0048046, -0.0001557, 0.0001481
7: -0.0144146, -0.0118163, -0.0144146, -0.0118163, -0.0010267, 0.0010790
8: 0.0052933, 0.0073546, 0.0052933, 0.0073546, -0.0008145, 0.0008561
9: 0.0072451, 0.0109526, 0.0072451, 0.0109526, -0.0014650, 0.0015397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 195
type: RSZ, layer: 3, pos: 179
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 238

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004035, upper bound: 0.0003957
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004282, upper bound: 0.0003876
time: 0.88 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003519, upper bound: 0.0003840
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003387, upper bound: 0.0003889
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003614, upper bound: 0.0003745
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003509, upper bound: 0.0003894
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003300, upper bound: 0.0002779
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003037, upper bound: 0.0003193
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003566, upper bound: 0.0004290
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003749, upper bound: 0.0003980
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003599, upper bound: 0.0004161
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003521, upper bound: 0.0004245
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0002212, upper bound: 0.0002207
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0002211, upper bound: 0.0002211
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004108, upper bound: 0.0003942
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004226, upper bound: 0.0003834
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004159, upper bound: 0.0003906
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004062, upper bound: 0.0003994
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003726, upper bound: 0.0003923
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003750, upper bound: 0.0003715
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004178, upper bound: 0.0003825
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004125, upper bound: 0.0003826
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003906, upper bound: 0.0003905
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004094, upper bound: 0.0003855
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004017, upper bound: 0.0003802
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003861, upper bound: 0.0003898
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004218, upper bound: 0.0003922
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004112, upper bound: 0.0003955
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003218, upper bound: 0.0002991
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0002885, upper bound: 0.0003263
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0002909, upper bound: 0.0002613
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0002578, upper bound: 0.0002918
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0002950, upper bound: 0.0002529
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0002665, upper bound: 0.0002837
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004179, upper bound: 0.0003900
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004156, upper bound: 0.0003916
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003164, upper bound: 0.0002957
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0002881, upper bound: 0.0003264
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004159, upper bound: 0.0003988
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004072, upper bound: 0.0004049
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003918, upper bound: 0.0003977
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004304, upper bound: 0.0003910
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003568, upper bound: 0.0003848
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003618, upper bound: 0.0003835
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004020, upper bound: 0.0003973
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003893, upper bound: 0.0004024
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003203, upper bound: 0.0003731
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003449, upper bound: 0.0003492
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003509, upper bound: 0.0003677
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003495, upper bound: 0.0003731
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004146, upper bound: 0.0003939
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004386, upper bound: 0.0003680
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004016, upper bound: 0.0003827
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003958, upper bound: 0.0003849
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003872, upper bound: 0.0003574
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003870, upper bound: 0.0003449
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003951, upper bound: 0.0003992
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004230, upper bound: 0.0003942
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003251, upper bound: 0.0002923
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0002947, upper bound: 0.0003229
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003544, upper bound: 0.0004097
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003744, upper bound: 0.0003817
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003664, upper bound: 0.0003648
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003677, upper bound: 0.0003553
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004102, upper bound: 0.0004010
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004366, upper bound: 0.0003862
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003882, upper bound: 0.0003974
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003762, upper bound: 0.0004027
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003874, upper bound: 0.0003965
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003757, upper bound: 0.0004087
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0002841, upper bound: 0.0002624
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0002556, upper bound: 0.0002946
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003509, upper bound: 0.0003677
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003495, upper bound: 0.0003731
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004012, upper bound: 0.0004170
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003895, upper bound: 0.0004212
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003171, upper bound: 0.0002996
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0002866, upper bound: 0.0003289
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004359, upper bound: 0.0003905
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004161, upper bound: 0.0003910
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003737, upper bound: 0.0003829
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004017, upper bound: 0.0003653
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004094, upper bound: 0.0004010
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004345, upper bound: 0.0003943
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003908, upper bound: 0.0004081
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004176, upper bound: 0.0003896
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003975, upper bound: 0.0004060
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004112, upper bound: 0.0003880
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004199, upper bound: 0.0003842
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004161, upper bound: 0.0003844
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003906, upper bound: 0.0003965
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003886, upper bound: 0.0004001
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003876, upper bound: 0.0004390
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004054, upper bound: 0.0004120
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003850, upper bound: 0.0003748
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003713, upper bound: 0.0003844
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0002211, upper bound: 0.0002211
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0002208, upper bound: 0.0002212
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003940, upper bound: 0.0004185
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003855, upper bound: 0.0004382
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004021, upper bound: 0.0004015
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003995, upper bound: 0.0004223
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003873, upper bound: 0.0004085
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003895, upper bound: 0.0004026
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003749, upper bound: 0.0003774
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004160, upper bound: 0.0003744
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003979, upper bound: 0.0004068
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003883, upper bound: 0.0004198
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003642, upper bound: 0.0003664
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003676, upper bound: 0.0003617
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003878, upper bound: 0.0003881
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003796, upper bound: 0.0004053
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003463, upper bound: 0.0003729
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003508, upper bound: 0.0003718
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0003290, upper bound: 0.0002919
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0002952, upper bound: 0.0003180
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004035, upper bound: 0.0003957
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 2, lower bound: -0.0004282, upper bound: 0.0003876
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 2, lower bound: -0.0004047, upper bound: 0.0004184
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 2, lower bound: -0.0003949, upper bound: 0.0004351
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 2, lower bound: -0.0003968, upper bound: 0.0004416
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 2, lower bound: -0.0004057, upper bound: 0.0004258
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 2, lower bound: -0.0004016, upper bound: 0.0004349
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 2, lower bound: -0.0004131, upper bound: 0.0003980
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 2, lower bound: -0.0003910, upper bound: 0.0003692
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 2, lower bound: -0.0003922, upper bound: 0.0003651
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 2, lower bound: -0.0004311, upper bound: 0.0003994
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 2, lower bound: -0.0004274, upper bound: 0.0004112
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 2, lower bound: -0.0004328, upper bound: 0.0003886
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 2, lower bound: -0.0004278, upper bound: 0.0003904
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 2, lower bound: -0.0003970, upper bound: 0.0003696
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 2, lower bound: -0.0003967, upper bound: 0.0003603

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 2.93 + 597.36 = 600.28 seconds
