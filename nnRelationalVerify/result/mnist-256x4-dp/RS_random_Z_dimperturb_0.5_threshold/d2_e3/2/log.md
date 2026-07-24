## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0013


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0023560, 0.0023560)
1: (-0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0005871, 0.0005871)
2: (0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0031111, 0.0031111)
3: (-0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0014160, 0.0014160)
4: (0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0006022, 0.0006022)
5: (0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0039130, 0.0039130)
6: (-0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0009932, 0.0009932)
7: (-0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0025696, 0.0025696)
8: (-0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0013513, 0.0013513)
9: (-0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0015669, 0.0015669)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.28 + 1.65 = 2.92 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0015295, upper bound: 0.0015294

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014927, upper bound: 0.0014968
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014967, upper bound: 0.0014927
time: 0.77 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -0.0014927, upper bound: 0.0014968
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -0.0014967, upper bound: 0.0014927

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0023303, 0.0023334
1: -0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0005807, 0.0005814
2: 0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0030812, 0.0030772
3: -0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0014006, 0.0014024
4: 0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0005964, 0.0005956
5: 0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0038753, 0.0038702
6: -0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0009823, 0.0009836
7: -0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0025415, 0.0025449
8: -0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0013366, 0.0013383
9: -0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0015519, 0.0015498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014326, upper bound: 0.0014727
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014676, upper bound: 0.0014327
time: 0.76 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0023334, 0.0023303
1: -0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0005814, 0.0005807
2: 0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0030772, 0.0030812
3: -0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0014024, 0.0014006
4: 0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0005956, 0.0005964
5: 0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0038702, 0.0038753
6: -0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0009836, 0.0009823
7: -0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0025449, 0.0025415
8: -0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0013383, 0.0013366
9: -0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0015498, 0.0015519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011193, upper bound: 0.0011185
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011193, upper bound: 0.0011185
time: 0.68 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.53 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -0.0014326, upper bound: 0.0014727
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -0.0014676, upper bound: 0.0014327
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.53
Output dim: 0, lower bound: -0.0011193, upper bound: 0.0011185
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.53
Output dim: 0, lower bound: -0.0011193, upper bound: 0.0011185

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0022401, 0.0022716
1: -0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0005582, 0.0005660
2: 0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0029996, 0.0029580
3: -0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0013464, 0.0013653
4: 0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0005806, 0.0005725
5: 0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0037728, 0.0037204
6: -0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0009443, 0.0009576
7: -0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0024432, 0.0024775
8: -0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0012848, 0.0013029
9: -0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0015108, 0.0014898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013829, upper bound: 0.0013750
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013395, upper bound: 0.0014242
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0022684, 0.0022432
1: -0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0005652, 0.0005589
2: 0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0029621, 0.0029954
3: -0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0013634, 0.0013482
4: 0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0005733, 0.0005797
5: 0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0037255, 0.0037674
6: -0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0009562, 0.0009456
7: -0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0024740, 0.0024465
8: -0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0013010, 0.0012866
9: -0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0014919, 0.0015086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014168, upper bound: 0.0013959
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014315, upper bound: 0.0013874
time: 0.76 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.72 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.0013829, upper bound: 0.0013750
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.0013395, upper bound: 0.0014242
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.0014168, upper bound: 0.0013959
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.0014315, upper bound: 0.0013874

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0021057, 0.0020917
1: -0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0005247, 0.0005212
2: 0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0027620, 0.0027805
3: -0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0012656, 0.0012571
4: 0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0005346, 0.0005382
5: 0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0034739, 0.0034972
6: -0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0008876, 0.0008817
7: -0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0022966, 0.0022812
8: -0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0012077, 0.0011997
9: -0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0013911, 0.0014004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013242, upper bound: 0.0013165
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013240, upper bound: 0.0013188
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0020602, 0.0021362
1: -0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0005133, 0.0005323
2: 0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0028209, 0.0027204
3: -0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0012382, 0.0012839
4: 0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0005460, 0.0005265
5: 0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0035479, 0.0034216
6: -0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0008684, 0.0009005
7: -0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0022469, 0.0023299
8: -0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0011816, 0.0012252
9: -0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0014207, 0.0013701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008849, upper bound: 0.0008863
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008849, upper bound: 0.0008863
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0022121, 0.0021967
1: -0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0005512, 0.0005474
2: 0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0029007, 0.0029211
3: -0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0013295, 0.0013203
4: 0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0005614, 0.0005654
5: 0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0036483, 0.0036739
6: -0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0009325, 0.0009260
7: -0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0024126, 0.0023958
8: -0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0012688, 0.0012599
9: -0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0014609, 0.0014712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013784, upper bound: 0.0013503
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013686, upper bound: 0.0013581
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0022228, 0.0021869
1: -0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0005539, 0.0005449
2: 0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0028878, 0.0029352
3: -0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0013360, 0.0013144
4: 0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0005589, 0.0005681
5: 0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0036321, 0.0036917
6: -0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0009370, 0.0009219
7: -0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0024243, 0.0023851
8: -0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0012749, 0.0012543
9: -0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0014544, 0.0014783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013759, upper bound: 0.0013287
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013745, upper bound: 0.0013288
time: 0.93 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -0.0013242, upper bound: 0.0013165
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -0.0013240, upper bound: 0.0013188
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -0.0008849, upper bound: 0.0008863
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -0.0008849, upper bound: 0.0008863
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -0.0013784, upper bound: 0.0013503
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -0.0013686, upper bound: 0.0013581
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -0.0013759, upper bound: 0.0013287
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -0.0013745, upper bound: 0.0013288

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0020238, 0.0020039
1: -0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0005043, 0.0004993
2: 0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0026461, 0.0026724
3: -0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0012164, 0.0012044
4: 0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0005122, 0.0005172
5: 0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0033282, 0.0033612
6: -0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0008531, 0.0008447
7: -0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0022072, 0.0021856
8: -0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0011608, 0.0011494
9: -0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0013327, 0.0013460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012864, upper bound: 0.0012814
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012767, upper bound: 0.0012818
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0020180, 0.0020078
1: -0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0005028, 0.0005003
2: 0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0026512, 0.0026647
3: -0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0012128, 0.0012067
4: 0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0005131, 0.0005157
5: 0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0033346, 0.0033515
6: -0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0008506, 0.0008463
7: -0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0022009, 0.0021898
8: -0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0011574, 0.0011516
9: -0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0013353, 0.0013421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012799, upper bound: 0.0012854
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012881, upper bound: 0.0012642
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0021531, 0.0021289
1: -0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0005365, 0.0005305
2: 0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0028112, 0.0028432
3: -0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0012941, 0.0012795
4: 0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0005441, 0.0005503
5: 0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0035357, 0.0035760
6: -0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0009076, 0.0008974
7: -0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0023483, 0.0023219
8: -0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0012350, 0.0012211
9: -0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0014159, 0.0014320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013225, upper bound: 0.0012934
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013215, upper bound: 0.0012940
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0021436, 0.0021377
1: -0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0005341, 0.0005327
2: 0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0028228, 0.0028307
3: -0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0012884, 0.0012848
4: 0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0005464, 0.0005479
5: 0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0035504, 0.0035602
6: -0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0009036, 0.0009011
7: -0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0023379, 0.0023315
8: -0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0012295, 0.0012261
9: -0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0014217, 0.0014257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012035, upper bound: 0.0011831
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012035, upper bound: 0.0011831
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0021396, 0.0021016
1: -0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0005331, 0.0005237
2: 0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0027751, 0.0028253
3: -0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0012859, 0.0012631
4: 0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0005371, 0.0005468
5: 0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0034903, 0.0035535
6: -0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0009019, 0.0008859
7: -0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0023335, 0.0022920
8: -0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0012272, 0.0012054
9: -0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0013977, 0.0014230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013260, upper bound: 0.0012386
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012815, upper bound: 0.0012794
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0021375, 0.0021033
1: -0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0005326, 0.0005241
2: 0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0027774, 0.0028225
3: -0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0012847, 0.0012641
4: 0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0005376, 0.0005463
5: 0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0034932, 0.0035499
6: -0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0009010, 0.0008866
7: -0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0023312, 0.0022939
8: -0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0012260, 0.0012064
9: -0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0013988, 0.0014216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007838, upper bound: 0.0007839
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007838, upper bound: 0.0007839
time: 0.68 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.57 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0012864, upper bound: 0.0012814
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0012767, upper bound: 0.0012818
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0012799, upper bound: 0.0012854
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0012881, upper bound: 0.0012642
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0013225, upper bound: 0.0012934
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0013215, upper bound: 0.0012940
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0012035, upper bound: 0.0011831
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0012035, upper bound: 0.0011831
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0013260, upper bound: 0.0012386
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0012815, upper bound: 0.0012794
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0007838, upper bound: 0.0007839
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0007838, upper bound: 0.0007839

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0020720, 0.0020443
1: -0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0005163, 0.0005094
2: 0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0026995, 0.0027360
3: -0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0012453, 0.0012287
4: 0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0005225, 0.0005295
5: 0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0033953, 0.0034412
6: -0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0008734, 0.0008618
7: -0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0022598, 0.0022296
8: -0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0011884, 0.0011725
9: -0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0013596, 0.0013780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007490, upper bound: 0.0007490
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007490, upper bound: 0.0007490
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0020686, 0.0020472
1: -0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0005154, 0.0005101
2: 0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0027033, 0.0027315
3: -0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0012433, 0.0012304
4: 0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0005232, 0.0005287
5: 0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0034001, 0.0034355
6: -0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0008720, 0.0008630
7: -0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0022561, 0.0022328
8: -0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0011864, 0.0011742
9: -0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0013615, 0.0013757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012716, upper bound: 0.0012117
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012269, upper bound: 0.0012407
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0020038, 0.0019192
1: -0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0004993, 0.0004782
2: 0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0025343, 0.0026460
3: -0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0012043, 0.0011535
4: 0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0004905, 0.0005121
5: 0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0031875, 0.0033279
6: -0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0008447, 0.0008090
7: -0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0021854, 0.0020932
8: -0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0011493, 0.0011008
9: -0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0012764, 0.0013326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013000, upper bound: 0.0012053
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012960, upper bound: 0.0012146
time: 0.95 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.98 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.98
Output dim: 0, lower bound: -0.0007490, upper bound: 0.0007490
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.98
Output dim: 0, lower bound: -0.0007490, upper bound: 0.0007490
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.98
Output dim: 0, lower bound: -0.0012716, upper bound: 0.0012117
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.98
Output dim: 0, lower bound: -0.0012269, upper bound: 0.0012407
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -0.0013000, upper bound: 0.0012053
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.98
Output dim: 0, lower bound: -0.0012960, upper bound: 0.0012146

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9895469, 0.9934982, 0.9895469, 0.9934982, -0.0019827, 0.0018979
1: -0.0038686, -0.0028840, -0.0038686, -0.0028840, -0.0004940, 0.0004729
2: 0.0052298, 0.0104475, 0.0052298, 0.0104475, -0.0025062, 0.0026181
3: -0.0060284, -0.0036535, -0.0060284, -0.0036535, -0.0011917, 0.0011407
4: 0.0015401, 0.0025500, 0.0015401, 0.0025500, -0.0004851, 0.0005067
5: 0.0055372, 0.0120997, 0.0055372, 0.0120997, -0.0031521, 0.0032929
6: -0.0015302, 0.0001354, -0.0015302, 0.0001354, -0.0008358, 0.0008000
7: -0.0070967, -0.0027872, -0.0070967, -0.0027872, -0.0021624, 0.0020699
8: -0.0032962, -0.0010299, -0.0032962, -0.0010299, -0.0011372, 0.0010886
9: -0.0006696, 0.0019583, -0.0006696, 0.0019583, -0.0012622, 0.0013186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012607, upper bound: 0.0011681
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012490, upper bound: 0.0011689
time: 0.90 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.00 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.00
Output dim: 0, lower bound: -0.0012607, upper bound: 0.0011681
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.00
Output dim: 0, lower bound: -0.0012490, upper bound: 0.0011689

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.92 + 52.48 = 55.41 seconds
