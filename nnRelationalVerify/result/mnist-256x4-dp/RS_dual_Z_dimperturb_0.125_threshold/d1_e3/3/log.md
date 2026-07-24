## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0028484


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0028148, -0.0022019, -0.0028148, -0.0022019, -0.0002469, 0.0002469)
1: (0.0240700, 0.0267277, 0.0240700, 0.0267277, -0.0012366, 0.0012366)
2: (0.0235038, 0.0252357, 0.0235038, 0.0252357, -0.0008220, 0.0008220)
3: (0.0113284, 0.0133113, 0.0113284, 0.0133113, -0.0010417, 0.0010417)
4: (-0.0135337, -0.0114182, -0.0135337, -0.0114182, -0.0010899, 0.0010899)
5: (0.0186529, 0.0210781, 0.0186529, 0.0210781, -0.0012814, 0.0012814)
6: (0.0092373, 0.0111440, 0.0092373, 0.0111440, -0.0010006, 0.0010006)
7: (-0.0183113, -0.0163864, -0.0183113, -0.0163864, -0.0009387, 0.0009387)
8: (0.0132910, 0.0152261, 0.0132910, 0.0152261, -0.0010475, 0.0010475)
9: (0.9195744, 0.9288392, 0.9195744, 0.9288392, -0.0048321, 0.0048321)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.37 + 1.24 = 2.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0033814, upper bound: 0.0033814

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0032780, upper bound: 0.0032371
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0032371, upper bound: 0.0032780
time: 0.41 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.96 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.96
Output dim: 9, lower bound: -0.0032780, upper bound: 0.0032371
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.96
Output dim: 9, lower bound: -0.0032371, upper bound: 0.0032780

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028148, -0.0022019, -0.0028148, -0.0022019, -0.0002467, 0.0002467
1: 0.0240700, 0.0267277, 0.0240700, 0.0267277, -0.0011670, 0.0011756
2: 0.0235038, 0.0252357, 0.0235038, 0.0252357, -0.0007899, 0.0007914
3: 0.0113284, 0.0133113, 0.0113284, 0.0133113, -0.0009825, 0.0009860
4: -0.0135337, -0.0114182, -0.0135337, -0.0114182, -0.0010550, 0.0010533
5: 0.0186529, 0.0210781, 0.0186529, 0.0210781, -0.0012310, 0.0012327
6: 0.0092373, 0.0111440, 0.0092373, 0.0111440, -0.0009589, 0.0009606
7: -0.0183113, -0.0163864, -0.0183113, -0.0163864, -0.0009079, 0.0009071
8: 0.0132910, 0.0152261, 0.0132910, 0.0152261, -0.0009866, 0.0009875
9: 0.9195744, 0.9288392, 0.9195744, 0.9288392, -0.0046042, 0.0045892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0026789, upper bound: 0.0028494
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0029131, upper bound: 0.0026414
time: 0.42 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028148, -0.0022019, -0.0028148, -0.0022019, -0.0002467, 0.0002467
1: 0.0240700, 0.0267277, 0.0240700, 0.0267277, -0.0011756, 0.0011670
2: 0.0235038, 0.0252357, 0.0235038, 0.0252357, -0.0007914, 0.0007899
3: 0.0113284, 0.0133113, 0.0113284, 0.0133113, -0.0009860, 0.0009825
4: -0.0135337, -0.0114182, -0.0135337, -0.0114182, -0.0010533, 0.0010550
5: 0.0186529, 0.0210781, 0.0186529, 0.0210781, -0.0012327, 0.0012310
6: 0.0092373, 0.0111440, 0.0092373, 0.0111440, -0.0009606, 0.0009589
7: -0.0183113, -0.0163864, -0.0183113, -0.0163864, -0.0009071, 0.0009079
8: 0.0132910, 0.0152261, 0.0132910, 0.0152261, -0.0009875, 0.0009866
9: 0.9195744, 0.9288392, 0.9195744, 0.9288392, -0.0045892, 0.0046042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0026414, upper bound: 0.0029131
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0028494, upper bound: 0.0026789
time: 0.42 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 9, lower bound: -0.0026789, upper bound: 0.0028494
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 9, lower bound: -0.0029131, upper bound: 0.0026414
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 9, lower bound: -0.0026414, upper bound: 0.0029131
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 9, lower bound: -0.0028494, upper bound: 0.0026789

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028148, -0.0022019, -0.0028148, -0.0022019, -0.0002347, 0.0002327
1: 0.0240700, 0.0267277, 0.0240700, 0.0267277, -0.0010839, 0.0010385
2: 0.0235038, 0.0252357, 0.0235038, 0.0252357, -0.0007371, 0.0006921
3: 0.0113284, 0.0133113, 0.0113284, 0.0133113, -0.0009042, 0.0008698
4: -0.0135337, -0.0114182, -0.0135337, -0.0114182, -0.0009325, 0.0009891
5: 0.0186529, 0.0210781, 0.0186529, 0.0210781, -0.0011456, 0.0010924
6: 0.0092373, 0.0111440, 0.0092373, 0.0111440, -0.0008925, 0.0008492
7: -0.0183113, -0.0163864, -0.0183113, -0.0163864, -0.0007924, 0.0008492
8: 0.0132910, 0.0152261, 0.0132910, 0.0152261, -0.0009069, 0.0008797
9: 0.9195744, 0.9288392, 0.9195744, 0.9288392, -0.0040662, 0.0042471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024593, upper bound: 0.0024492
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0023153, upper bound: 0.0026579
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028148, -0.0022019, -0.0028148, -0.0022019, -0.0002327, 0.0002467
1: 0.0240700, 0.0267277, 0.0240700, 0.0267277, -0.0010299, 0.0011756
2: 0.0235038, 0.0252357, 0.0235038, 0.0252357, -0.0006906, 0.0007914
3: 0.0113284, 0.0133113, 0.0113284, 0.0133113, -0.0008663, 0.0009860
4: -0.0135337, -0.0114182, -0.0135337, -0.0114182, -0.0010550, 0.0009308
5: 0.0186529, 0.0210781, 0.0186529, 0.0210781, -0.0010907, 0.0012327
6: 0.0092373, 0.0111440, 0.0092373, 0.0111440, -0.0008474, 0.0009606
7: -0.0183113, -0.0163864, -0.0183113, -0.0163864, -0.0009079, 0.0007916
8: 0.0132910, 0.0152261, 0.0132910, 0.0152261, -0.0008788, 0.0009875
9: 0.9195744, 0.9288392, 0.9195744, 0.9288392, -0.0046042, 0.0040513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0027311, upper bound: 0.0022998
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024602, upper bound: 0.0024128
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028148, -0.0022019, -0.0028148, -0.0022019, -0.0002347, 0.0002327
1: 0.0240700, 0.0267277, 0.0240700, 0.0267277, -0.0010896, 0.0010299
2: 0.0235038, 0.0252357, 0.0235038, 0.0252357, -0.0007407, 0.0006906
3: 0.0113284, 0.0133113, 0.0113284, 0.0133113, -0.0009135, 0.0008663
4: -0.0135337, -0.0114182, -0.0135337, -0.0114182, -0.0009308, 0.0009896
5: 0.0186529, 0.0210781, 0.0186529, 0.0210781, -0.0011481, 0.0010907
6: 0.0092373, 0.0111440, 0.0092373, 0.0111440, -0.0008943, 0.0008474
7: -0.0183113, -0.0163864, -0.0183113, -0.0163864, -0.0007916, 0.0008513
8: 0.0132910, 0.0152261, 0.0132910, 0.0152261, -0.0009161, 0.0008788
9: 0.9195744, 0.9288392, 0.9195744, 0.9288392, -0.0040513, 0.0042783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024128, upper bound: 0.0024602
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0022998, upper bound: 0.0027311
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028148, -0.0022019, -0.0028148, -0.0022019, -0.0002327, 0.0002467
1: 0.0240700, 0.0267277, 0.0240700, 0.0267277, -0.0010385, 0.0011670
2: 0.0235038, 0.0252357, 0.0235038, 0.0252357, -0.0006921, 0.0007899
3: 0.0113284, 0.0133113, 0.0113284, 0.0133113, -0.0008698, 0.0009825
4: -0.0135337, -0.0114182, -0.0135337, -0.0114182, -0.0010533, 0.0009325
5: 0.0186529, 0.0210781, 0.0186529, 0.0210781, -0.0010924, 0.0012310
6: 0.0092373, 0.0111440, 0.0092373, 0.0111440, -0.0008492, 0.0009589
7: -0.0183113, -0.0163864, -0.0183113, -0.0163864, -0.0009071, 0.0007924
8: 0.0132910, 0.0152261, 0.0132910, 0.0152261, -0.0008797, 0.0009866
9: 0.9195744, 0.9288392, 0.9195744, 0.9288392, -0.0045892, 0.0040662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0026579, upper bound: 0.0023153
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0024492, upper bound: 0.0024593
time: 0.43 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.30 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.30
Output dim: 9, lower bound: -0.0024593, upper bound: 0.0024492
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.30
Output dim: 9, lower bound: -0.0023153, upper bound: 0.0026579
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.30
Output dim: 9, lower bound: -0.0027311, upper bound: 0.0022998
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.30
Output dim: 9, lower bound: -0.0024602, upper bound: 0.0024128
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.30
Output dim: 9, lower bound: -0.0024128, upper bound: 0.0024602
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.30
Output dim: 9, lower bound: -0.0022998, upper bound: 0.0027311
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.30
Output dim: 9, lower bound: -0.0026579, upper bound: 0.0023153
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.30
Output dim: 9, lower bound: -0.0024492, upper bound: 0.0024593

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.61 + 14.72 = 17.33 seconds
