## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00379488


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0003228, 0.0001532, -0.0003228, 0.0001532, -0.0002752, 0.0002752)
1: (-0.0001909, 0.0015565, -0.0001909, 0.0015565, -0.0009403, 0.0009403)
2: (0.0140090, 0.0166259, 0.0140090, 0.0166259, -0.0013714, 0.0013714)
3: (-0.0000927, 0.0018751, -0.0000927, 0.0018751, -0.0010153, 0.0010153)
4: (-0.0044651, -0.0026501, -0.0044651, -0.0026501, -0.0010723, 0.0010723)
5: (0.0078456, 0.0098098, 0.0078456, 0.0098098, -0.0010120, 0.0010120)
6: (0.0092310, 0.0099722, 0.0092310, 0.0099722, -0.0007383, 0.0007383)
7: (-0.0196955, -0.0154315, -0.0196955, -0.0154315, -0.0020843, 0.0020843)
8: (0.9673610, 0.9795778, 0.9673610, 0.9795778, -0.0064393, 0.0064393)
9: (0.0036552, 0.0072458, 0.0036552, 0.0072458, -0.0017843, 0.0017843)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.54 + 1.28 = 2.82 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0050421, upper bound: 0.0050421

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049151, upper bound: 0.0048826
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048826, upper bound: 0.0049151
time: 0.45 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.91 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.91
Output dim: 8, lower bound: -0.0049151, upper bound: 0.0048826
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.91
Output dim: 8, lower bound: -0.0048826, upper bound: 0.0049151

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0003228, 0.0001532, -0.0003228, 0.0001532, -0.0002709, 0.0002702
1: -0.0001909, 0.0015565, -0.0001909, 0.0015565, -0.0009115, 0.0009082
2: 0.0140090, 0.0166259, 0.0140090, 0.0166259, -0.0013233, 0.0013284
3: -0.0000927, 0.0018751, -0.0000927, 0.0018751, -0.0009791, 0.0009829
4: -0.0044651, -0.0026501, -0.0044651, -0.0026501, -0.0010389, 0.0010425
5: 0.0078456, 0.0098098, 0.0078456, 0.0098098, -0.0009759, 0.0009797
6: 0.0092310, 0.0099722, 0.0092310, 0.0099722, -0.0007261, 0.0007246
7: -0.0196955, -0.0154315, -0.0196955, -0.0154315, -0.0020142, 0.0020060
8: 0.9673610, 0.9795778, 0.9673610, 0.9795778, -0.0062384, 0.0062148
9: 0.0036552, 0.0072458, 0.0036552, 0.0072458, -0.0017183, 0.0017253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041824, upper bound: 0.0041824
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041824, upper bound: 0.0041824
time: 0.45 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0003228, 0.0001532, -0.0003228, 0.0001532, -0.0002702, 0.0002752
1: -0.0001909, 0.0015565, -0.0001909, 0.0015565, -0.0009082, 0.0009403
2: 0.0140090, 0.0166259, 0.0140090, 0.0166259, -0.0013714, 0.0013233
3: -0.0000927, 0.0018751, -0.0000927, 0.0018751, -0.0010153, 0.0009791
4: -0.0044651, -0.0026501, -0.0044651, -0.0026501, -0.0010723, 0.0010389
5: 0.0078456, 0.0098098, 0.0078456, 0.0098098, -0.0010120, 0.0009759
6: 0.0092310, 0.0099722, 0.0092310, 0.0099722, -0.0007246, 0.0007383
7: -0.0196955, -0.0154315, -0.0196955, -0.0154315, -0.0020060, 0.0020843
8: 0.9673610, 0.9795778, 0.9673610, 0.9795778, -0.0062148, 0.0064393
9: 0.0036552, 0.0072458, 0.0036552, 0.0072458, -0.0017843, 0.0017183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041204, upper bound: 0.0045906
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045719, upper bound: 0.0041412
time: 0.44 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 8, lower bound: -0.0041824, upper bound: 0.0041824
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 8, lower bound: -0.0041824, upper bound: 0.0041824
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 8, lower bound: -0.0041204, upper bound: 0.0045906
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 8, lower bound: -0.0045719, upper bound: 0.0041412

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0003228, 0.0001532, -0.0003228, 0.0001532, -0.0002710, 0.0002697
1: -0.0001909, 0.0015565, -0.0001909, 0.0015565, -0.0009124, 0.0009063
2: 0.0140090, 0.0166259, 0.0140090, 0.0166259, -0.0013205, 0.0013296
3: -0.0000927, 0.0018751, -0.0000927, 0.0018751, -0.0009770, 0.0009838
4: -0.0044651, -0.0026501, -0.0044651, -0.0026501, -0.0010370, 0.0010433
5: 0.0078456, 0.0098098, 0.0078456, 0.0098098, -0.0009738, 0.0009806
6: 0.0092310, 0.0099722, 0.0092310, 0.0099722, -0.0007264, 0.0007238
7: -0.0196955, -0.0154315, -0.0196955, -0.0154315, -0.0020162, 0.0020013
8: 0.9673610, 0.9795778, 0.9673610, 0.9795778, -0.0062441, 0.0062015
9: 0.0036552, 0.0072458, 0.0036552, 0.0072458, -0.0017144, 0.0017270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037333, upper bound: 0.0037333
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037333, upper bound: 0.0037333
time: 0.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0003228, 0.0001532, -0.0003228, 0.0001532, -0.0002704, 0.0002702
1: -0.0001909, 0.0015565, -0.0001909, 0.0015565, -0.0009096, 0.0009082
2: 0.0140090, 0.0166259, 0.0140090, 0.0166259, -0.0013233, 0.0013256
3: -0.0000927, 0.0018751, -0.0000927, 0.0018751, -0.0009791, 0.0009808
4: -0.0044651, -0.0026501, -0.0044651, -0.0026501, -0.0010389, 0.0010405
5: 0.0078456, 0.0098098, 0.0078456, 0.0098098, -0.0009759, 0.0009775
6: 0.0092310, 0.0099722, 0.0092310, 0.0099722, -0.0007253, 0.0007246
7: -0.0196955, -0.0154315, -0.0196955, -0.0154315, -0.0020096, 0.0020060
8: 0.9673610, 0.9795778, 0.9673610, 0.9795778, -0.0062251, 0.0062148
9: 0.0036552, 0.0072458, 0.0036552, 0.0072458, -0.0017183, 0.0017214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0027109, upper bound: 0.0027109
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0027109, upper bound: 0.0027109
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0003228, 0.0001532, -0.0003228, 0.0001532, -0.0002396, 0.0002566
1: -0.0001909, 0.0015565, -0.0001909, 0.0015565, -0.0007994, 0.0008872
2: 0.0140090, 0.0166259, 0.0140090, 0.0166259, -0.0012834, 0.0011518
3: -0.0000927, 0.0018751, -0.0000927, 0.0018751, -0.0009448, 0.0008458
4: -0.0044651, -0.0026501, -0.0044651, -0.0026501, -0.0010322, 0.0009409
5: 0.0078456, 0.0098098, 0.0078456, 0.0098098, -0.0009411, 0.0008424
6: 0.0092310, 0.0099722, 0.0092310, 0.0099722, -0.0006977, 0.0007350
7: -0.0196955, -0.0154315, -0.0196955, -0.0154315, -0.0016575, 0.0018719
8: 0.9673610, 0.9795778, 0.9673610, 0.9795778, -0.0054234, 0.0060377
9: 0.0036552, 0.0072458, 0.0036552, 0.0072458, -0.0016163, 0.0014357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040130, upper bound: 0.0045608
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040890, upper bound: 0.0045091
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0003228, 0.0001532, -0.0003228, 0.0001532, -0.0002516, 0.0002447
1: -0.0001909, 0.0015565, -0.0001909, 0.0015565, -0.0008551, 0.0008315
2: 0.0140090, 0.0166259, 0.0140090, 0.0166259, -0.0011999, 0.0012353
3: -0.0000927, 0.0018751, -0.0000927, 0.0018751, -0.0008820, 0.0009086
4: -0.0044651, -0.0026501, -0.0044651, -0.0026501, -0.0009743, 0.0009988
5: 0.0078456, 0.0098098, 0.0078456, 0.0098098, -0.0008785, 0.0009050
6: 0.0092310, 0.0099722, 0.0092310, 0.0099722, -0.0007213, 0.0007113
7: -0.0196955, -0.0154315, -0.0196955, -0.0154315, -0.0017935, 0.0017358
8: 0.9673610, 0.9795778, 0.9673610, 0.9795778, -0.0058132, 0.0056479
9: 0.0036552, 0.0072458, 0.0036552, 0.0072458, -0.0015017, 0.0015503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044889, upper bound: 0.0041131
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045412, upper bound: 0.0040683
time: 0.44 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.24
Output dim: 8, lower bound: -0.0037333, upper bound: 0.0037333
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.24
Output dim: 8, lower bound: -0.0037333, upper bound: 0.0037333
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.24
Output dim: 8, lower bound: -0.0027109, upper bound: 0.0027109
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.24
Output dim: 8, lower bound: -0.0027109, upper bound: 0.0027109
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 8, lower bound: -0.0040130, upper bound: 0.0045608
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 8, lower bound: -0.0040890, upper bound: 0.0045091
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 8, lower bound: -0.0044889, upper bound: 0.0041131
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 8, lower bound: -0.0045412, upper bound: 0.0040683

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0003228, 0.0001532, -0.0003228, 0.0001532, -0.0002354, 0.0002538
1: -0.0001909, 0.0015565, -0.0001909, 0.0015565, -0.0007708, 0.0008616
2: 0.0140090, 0.0166259, 0.0140090, 0.0166259, -0.0012426, 0.0011068
3: -0.0000927, 0.0018751, -0.0000927, 0.0018751, -0.0009131, 0.0008109
4: -0.0044651, -0.0026501, -0.0044651, -0.0026501, -0.0010076, 0.0009133
5: 0.0078456, 0.0098098, 0.0078456, 0.0098098, -0.0009094, 0.0008075
6: 0.0092310, 0.0099722, 0.0092310, 0.0099722, -0.0006860, 0.0007245
7: -0.0196955, -0.0154315, -0.0196955, -0.0154315, -0.0015814, 0.0018028
8: 0.9673610, 0.9795778, 0.9673610, 0.9795778, -0.0052163, 0.0058506
9: 0.0036552, 0.0072458, 0.0036552, 0.0072458, -0.0015588, 0.0013723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039521, upper bound: 0.0044271
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0038591, upper bound: 0.0044936
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0003228, 0.0001532, -0.0003228, 0.0001532, -0.0002368, 0.0002530
1: -0.0001909, 0.0015565, -0.0001909, 0.0015565, -0.0007773, 0.0008580
2: 0.0140090, 0.0166259, 0.0140090, 0.0166259, -0.0012373, 0.0011165
3: -0.0000927, 0.0018751, -0.0000927, 0.0018751, -0.0009091, 0.0008182
4: -0.0044651, -0.0026501, -0.0044651, -0.0026501, -0.0010039, 0.0009201
5: 0.0078456, 0.0098098, 0.0078456, 0.0098098, -0.0009055, 0.0008148
6: 0.0092310, 0.0099722, 0.0092310, 0.0099722, -0.0006887, 0.0007230
7: -0.0196955, -0.0154315, -0.0196955, -0.0154315, -0.0015973, 0.0017941
8: 0.9673610, 0.9795778, 0.9673610, 0.9795778, -0.0052617, 0.0058258
9: 0.0036552, 0.0072458, 0.0036552, 0.0072458, -0.0015515, 0.0013857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026811, upper bound: 0.0026803
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026811, upper bound: 0.0026803
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0003228, 0.0001532, -0.0003228, 0.0001532, -0.0002480, 0.0002419
1: -0.0001909, 0.0015565, -0.0001909, 0.0015565, -0.0008295, 0.0008058
2: 0.0140090, 0.0166259, 0.0140090, 0.0166259, -0.0011592, 0.0011947
3: -0.0000927, 0.0018751, -0.0000927, 0.0018751, -0.0008503, 0.0008770
4: -0.0044651, -0.0026501, -0.0044651, -0.0026501, -0.0009497, 0.0009743
5: 0.0078456, 0.0098098, 0.0078456, 0.0098098, -0.0008468, 0.0008734
6: 0.0092310, 0.0099722, 0.0092310, 0.0099722, -0.0007109, 0.0007008
7: -0.0196955, -0.0154315, -0.0196955, -0.0154315, -0.0017246, 0.0016667
8: 0.9673610, 0.9795778, 0.9673610, 0.9795778, -0.0056267, 0.0054608
9: 0.0036552, 0.0072458, 0.0036552, 0.0072458, -0.0014442, 0.0014930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026803, upper bound: 0.0026811
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026803, upper bound: 0.0026811
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0003228, 0.0001532, -0.0003228, 0.0001532, -0.0002487, 0.0002405
1: -0.0001909, 0.0015565, -0.0001909, 0.0015565, -0.0008331, 0.0007993
2: 0.0140090, 0.0166259, 0.0140090, 0.0166259, -0.0011494, 0.0012000
3: -0.0000927, 0.0018751, -0.0000927, 0.0018751, -0.0008430, 0.0008810
4: -0.0044651, -0.0026501, -0.0044651, -0.0026501, -0.0009429, 0.0009780
5: 0.0078456, 0.0098098, 0.0078456, 0.0098098, -0.0008395, 0.0008774
6: 0.0092310, 0.0099722, 0.0092310, 0.0099722, -0.0007124, 0.0006980
7: -0.0196955, -0.0154315, -0.0196955, -0.0154315, -0.0017333, 0.0016509
8: 0.9673610, 0.9795778, 0.9673610, 0.9795778, -0.0056515, 0.0054154
9: 0.0036552, 0.0072458, 0.0036552, 0.0072458, -0.0014309, 0.0015003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044766, upper bound: 0.0039308
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044138, upper bound: 0.0039979
time: 0.46 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.30 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 8, lower bound: -0.0039521, upper bound: 0.0044271
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 8, lower bound: -0.0038591, upper bound: 0.0044936
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.30
Output dim: 8, lower bound: -0.0026811, upper bound: 0.0026803
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.30
Output dim: 8, lower bound: -0.0026811, upper bound: 0.0026803
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.30
Output dim: 8, lower bound: -0.0026803, upper bound: 0.0026811
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.30
Output dim: 8, lower bound: -0.0026803, upper bound: 0.0026811
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 8, lower bound: -0.0044766, upper bound: 0.0039308
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 8, lower bound: -0.0044138, upper bound: 0.0039979

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0003228, 0.0001532, -0.0003228, 0.0001532, -0.0002332, 0.0002506
1: -0.0001909, 0.0015565, -0.0001909, 0.0015565, -0.0007602, 0.0008473
2: 0.0140090, 0.0166259, 0.0140090, 0.0166259, -0.0012212, 0.0010908
3: -0.0000927, 0.0018751, -0.0000927, 0.0018751, -0.0008970, 0.0007989
4: -0.0044651, -0.0026501, -0.0044651, -0.0026501, -0.0009927, 0.0009022
5: 0.0078456, 0.0098098, 0.0078456, 0.0098098, -0.0008934, 0.0007955
6: 0.0092310, 0.0099722, 0.0092310, 0.0099722, -0.0006814, 0.0007184
7: -0.0196955, -0.0154315, -0.0196955, -0.0154315, -0.0015554, 0.0017679
8: 0.9673610, 0.9795778, 0.9673610, 0.9795778, -0.0051417, 0.0057506
9: 0.0036552, 0.0072458, 0.0036552, 0.0072458, -0.0015294, 0.0013504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0022037, upper bound: 0.0022132
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0022037, upper bound: 0.0022132
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0003228, 0.0001532, -0.0003228, 0.0001532, -0.0002323, 0.0002512
1: -0.0001909, 0.0015565, -0.0001909, 0.0015565, -0.0007558, 0.0008500
2: 0.0140090, 0.0166259, 0.0140090, 0.0166259, -0.0012254, 0.0010843
3: -0.0000927, 0.0018751, -0.0000927, 0.0018751, -0.0009001, 0.0007940
4: -0.0044651, -0.0026501, -0.0044651, -0.0026501, -0.0009956, 0.0008977
5: 0.0078456, 0.0098098, 0.0078456, 0.0098098, -0.0008965, 0.0007906
6: 0.0092310, 0.0099722, 0.0092310, 0.0099722, -0.0006796, 0.0007196
7: -0.0196955, -0.0154315, -0.0196955, -0.0154315, -0.0015447, 0.0017746
8: 0.9673610, 0.9795778, 0.9673610, 0.9795778, -0.0051111, 0.0057700
9: 0.0036552, 0.0072458, 0.0036552, 0.0072458, -0.0015351, 0.0013414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0022021, upper bound: 0.0022157
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0022021, upper bound: 0.0022157
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0003228, 0.0001532, -0.0003228, 0.0001532, -0.0002462, 0.0002373
1: -0.0001909, 0.0015565, -0.0001909, 0.0015565, -0.0008208, 0.0007850
2: 0.0140090, 0.0166259, 0.0140090, 0.0166259, -0.0011280, 0.0011816
3: -0.0000927, 0.0018751, -0.0000927, 0.0018751, -0.0008269, 0.0008672
4: -0.0044651, -0.0026501, -0.0044651, -0.0026501, -0.0009280, 0.0009652
5: 0.0078456, 0.0098098, 0.0078456, 0.0098098, -0.0008234, 0.0008636
6: 0.0092310, 0.0099722, 0.0092310, 0.0099722, -0.0007072, 0.0006920
7: -0.0196955, -0.0154315, -0.0196955, -0.0154315, -0.0017034, 0.0016160
8: 0.9673610, 0.9795778, 0.9673610, 0.9795778, -0.0055658, 0.0053154
9: 0.0036552, 0.0072458, 0.0036552, 0.0072458, -0.0014015, 0.0014751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0022157, upper bound: 0.0022021
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0022157, upper bound: 0.0022021
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0003228, 0.0001532, -0.0003228, 0.0001532, -0.0002456, 0.0002382
1: -0.0001909, 0.0015565, -0.0001909, 0.0015565, -0.0008180, 0.0007894
2: 0.0140090, 0.0166259, 0.0140090, 0.0166259, -0.0011345, 0.0011775
3: -0.0000927, 0.0018751, -0.0000927, 0.0018751, -0.0008318, 0.0008641
4: -0.0044651, -0.0026501, -0.0044651, -0.0026501, -0.0009326, 0.0009624
5: 0.0078456, 0.0098098, 0.0078456, 0.0098098, -0.0008283, 0.0008605
6: 0.0092310, 0.0099722, 0.0092310, 0.0099722, -0.0007060, 0.0006938
7: -0.0196955, -0.0154315, -0.0196955, -0.0154315, -0.0016966, 0.0016266
8: 0.9673610, 0.9795778, 0.9673610, 0.9795778, -0.0055464, 0.0053459
9: 0.0036552, 0.0072458, 0.0036552, 0.0072458, -0.0014104, 0.0014694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0022132, upper bound: 0.0022037
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0022132, upper bound: 0.0022037
time: 0.52 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.29 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 8, lower bound: -0.0022037, upper bound: 0.0022132
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 8, lower bound: -0.0022037, upper bound: 0.0022132
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 8, lower bound: -0.0022021, upper bound: 0.0022157
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 8, lower bound: -0.0022021, upper bound: 0.0022157
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 8, lower bound: -0.0022157, upper bound: 0.0022021
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 8, lower bound: -0.0022157, upper bound: 0.0022021
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 8, lower bound: -0.0022132, upper bound: 0.0022037
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 8, lower bound: -0.0022132, upper bound: 0.0022037

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.82 + 32.08 = 34.89 seconds
