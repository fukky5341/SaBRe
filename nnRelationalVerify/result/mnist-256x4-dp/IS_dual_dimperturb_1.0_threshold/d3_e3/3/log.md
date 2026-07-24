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
Threshold: 0.001979395


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078547, 0.0078547)
1: (-0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022145, 0.0022145)
2: (-0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0163393, 0.0163393)
3: (-0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021622, 0.0021622)
4: (0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0122110, 0.0122110)
5: (0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033926, 0.0033926)
6: (0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030794, 0.0030794)
7: (-0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114919, 0.0114919)
8: (-0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0089442, 0.0089442)
9: (-0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007717, 0.0007717)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.40 + 2.74 = 4.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0023286, upper bound: 0.0023287

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022467, upper bound: 0.0022188
time: 1.36 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022467, upper bound: 0.0022467
time: 1.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.13 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 3.13
Output dim: 5, lower bound: -0.0022467, upper bound: 0.0022188
IS_A2, status: Status.UNKNOWN, split count: 1, time: 3.13
Output dim: 5, lower bound: -0.0022467, upper bound: 0.0022467

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0152866, -0.0061908, -0.0155945, -0.0061038, -0.0074385, 0.0073441
1: -0.0072485, -0.0046841, -0.0073353, -0.0046596, -0.0020972, 0.0020706
2: -0.0149214, 0.0039997, -0.0155618, 0.0041807, -0.0154736, 0.0152772
3: -0.0003473, 0.0021566, -0.0004321, 0.0021805, -0.0020477, 0.0020217
4: 0.0031027, 0.0172432, 0.0029675, 0.0177218, -0.0114172, 0.0115640
5: 0.9963683, 1.0002968, 0.9963307, 1.0004299, -0.0031720, 0.0032128
6: 0.0045871, 0.0081532, 0.0045530, 0.0082739, -0.0028793, 0.0029163
7: -0.0062631, 0.0070446, -0.0063904, 0.0074951, -0.0107449, 0.0108830
8: -0.0146757, -0.0043183, -0.0150263, -0.0042192, -0.0084703, 0.0083627
9: -0.0036372, -0.0027436, -0.0036457, -0.0027133, -0.0007215, 0.0007308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022188, upper bound: 0.0022188
time: 1.89 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022188, upper bound: 0.0022187
time: 1.99 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0153697, -0.0059212, -0.0155960, -0.0061009, -0.0075136, 0.0080977
1: -0.0072719, -0.0046081, -0.0073357, -0.0046587, -0.0021184, 0.0022831
2: -0.0150942, 0.0045606, -0.0155649, 0.0041867, -0.0156297, 0.0168449
3: -0.0003702, 0.0022308, -0.0004325, 0.0021813, -0.0020683, 0.0022292
4: 0.0026835, 0.0173724, 0.0029630, 0.0177241, -0.0125888, 0.0116807
5: 0.9962518, 1.0003327, 0.9963294, 1.0004305, -0.0034976, 0.0032452
6: 0.0044814, 0.0081857, 0.0045519, 0.0082744, -0.0031747, 0.0029457
7: -0.0066576, 0.0071662, -0.0063947, 0.0074972, -0.0118475, 0.0109928
8: -0.0147703, -0.0040112, -0.0150280, -0.0042159, -0.0085557, 0.0092209
9: -0.0036637, -0.0027354, -0.0036460, -0.0027132, -0.0007955, 0.0007381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022188, upper bound: 0.0022467
time: 1.85 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022188, upper bound: 0.0022467
time: 2.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.31 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.31
Output dim: 5, lower bound: -0.0022188, upper bound: 0.0022188
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.31
Output dim: 5, lower bound: -0.0022188, upper bound: 0.0022187
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.31
Output dim: 5, lower bound: -0.0022188, upper bound: 0.0022467
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.31
Output dim: 5, lower bound: -0.0022188, upper bound: 0.0022467

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0152866, -0.0061908, -0.0152866, -0.0061908, -0.0070579, 0.0070579
1: -0.0072485, -0.0046841, -0.0072485, -0.0046841, -0.0019899, 0.0019899
2: -0.0149214, 0.0039997, -0.0149214, 0.0039997, -0.0146818, 0.0146818
3: -0.0003473, 0.0021566, -0.0003473, 0.0021566, -0.0019429, 0.0019429
4: 0.0031027, 0.0172432, 0.0031027, 0.0172432, -0.0109722, 0.0109722
5: 0.9963683, 1.0002968, 0.9963683, 1.0002968, -0.0030484, 0.0030484
6: 0.0045871, 0.0081532, 0.0045871, 0.0081532, -0.0027670, 0.0027670
7: -0.0062631, 0.0070446, -0.0062631, 0.0070446, -0.0103261, 0.0103261
8: -0.0146757, -0.0043183, -0.0146757, -0.0043183, -0.0080368, 0.0080368
9: -0.0036372, -0.0027436, -0.0036372, -0.0027436, -0.0006934, 0.0006934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021889, upper bound: 0.0021626
time: 1.51 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021888, upper bound: 0.0021889
time: 1.45 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0152866, -0.0061908, -0.0153697, -0.0059212, -0.0077822, 0.0072402
1: -0.0072485, -0.0046841, -0.0072719, -0.0046081, -0.0021941, 0.0020413
2: -0.0149214, 0.0039997, -0.0150942, 0.0045606, -0.0161886, 0.0150612
3: -0.0003473, 0.0021566, -0.0003702, 0.0022308, -0.0021423, 0.0019931
4: 0.0031027, 0.0172432, 0.0026835, 0.0173724, -0.0112558, 0.0120983
5: 0.9963683, 1.0002968, 0.9962518, 1.0003327, -0.0031272, 0.0033613
6: 0.0045871, 0.0081532, 0.0044814, 0.0081857, -0.0028385, 0.0030510
7: -0.0062631, 0.0070446, -0.0066576, 0.0071662, -0.0105930, 0.0113859
8: -0.0146757, -0.0043183, -0.0147703, -0.0040112, -0.0088617, 0.0082445
9: -0.0036372, -0.0027436, -0.0036637, -0.0027354, -0.0007113, 0.0007645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021634, upper bound: 0.0021889
time: 1.92 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021888, upper bound: 0.0021888
time: 2.06 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0153697, -0.0059212, -0.0152866, -0.0061908, -0.0072402, 0.0077822
1: -0.0072719, -0.0046081, -0.0072485, -0.0046841, -0.0020413, 0.0021941
2: -0.0150942, 0.0045606, -0.0149214, 0.0039997, -0.0150612, 0.0161886
3: -0.0003702, 0.0022308, -0.0003473, 0.0021566, -0.0019931, 0.0021423
4: 0.0026835, 0.0173724, 0.0031027, 0.0172432, -0.0120983, 0.0112558
5: 0.9962518, 1.0003327, 0.9963683, 1.0002968, -0.0033613, 0.0031272
6: 0.0044814, 0.0081857, 0.0045871, 0.0081532, -0.0030510, 0.0028385
7: -0.0066576, 0.0071662, -0.0062631, 0.0070446, -0.0113859, 0.0105930
8: -0.0147703, -0.0040112, -0.0146757, -0.0043183, -0.0082445, 0.0088617
9: -0.0036637, -0.0027354, -0.0036372, -0.0027436, -0.0007645, 0.0007113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021888, upper bound: 0.0021925
time: 1.96 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021887, upper bound: 0.0022156
time: 1.85 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0153697, -0.0059212, -0.0153697, -0.0059212, -0.0075718, 0.0075718
1: -0.0072719, -0.0046081, -0.0072719, -0.0046081, -0.0021348, 0.0021348
2: -0.0150942, 0.0045606, -0.0150942, 0.0045606, -0.0157508, 0.0157508
3: -0.0003702, 0.0022308, -0.0003702, 0.0022308, -0.0020844, 0.0020844
4: 0.0026835, 0.0173724, 0.0026835, 0.0173724, -0.0117712, 0.0117712
5: 0.9962518, 1.0003327, 0.9962518, 1.0003327, -0.0032704, 0.0032704
6: 0.0044814, 0.0081857, 0.0044814, 0.0081857, -0.0029685, 0.0029685
7: -0.0066576, 0.0071662, -0.0066576, 0.0071662, -0.0110780, 0.0110780
8: -0.0147703, -0.0040112, -0.0147703, -0.0040112, -0.0086220, 0.0086220
9: -0.0036637, -0.0027354, -0.0036637, -0.0027354, -0.0007439, 0.0007439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021888, upper bound: 0.0021926
time: 2.09 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021887, upper bound: 0.0022156
time: 1.51 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.04 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 5, lower bound: -0.0021889, upper bound: 0.0021626
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 5, lower bound: -0.0021888, upper bound: 0.0021889
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 5, lower bound: -0.0021634, upper bound: 0.0021889
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 5, lower bound: -0.0021888, upper bound: 0.0021888
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 5, lower bound: -0.0021888, upper bound: 0.0021925
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 5, lower bound: -0.0021887, upper bound: 0.0022156
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 5, lower bound: -0.0021888, upper bound: 0.0021926
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 5, lower bound: -0.0021887, upper bound: 0.0022156

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0150751, -0.0062721, -0.0152318, -0.0062112, -0.0068930, 0.0069319
1: -0.0071889, -0.0047070, -0.0072331, -0.0046898, -0.0019434, 0.0019544
2: -0.0144813, 0.0038306, -0.0148074, 0.0039573, -0.0143388, 0.0144198
3: -0.0002891, 0.0021342, -0.0003322, 0.0021510, -0.0018975, 0.0019082
4: 0.0032291, 0.0169143, 0.0031344, 0.0171580, -0.0107764, 0.0107159
5: 0.9964034, 1.0002056, 0.9963771, 1.0002732, -0.0029940, 0.0029772
6: 0.0046190, 0.0080702, 0.0045951, 0.0081317, -0.0027177, 0.0027024
7: -0.0061442, 0.0067351, -0.0062333, 0.0069644, -0.0101418, 0.0100849
8: -0.0144348, -0.0044109, -0.0146133, -0.0043415, -0.0078491, 0.0078934
9: -0.0036292, -0.0027644, -0.0036352, -0.0027490, -0.0006810, 0.0006772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0021024
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021556, upper bound: 0.0021286
time: 1.40 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0151588, -0.0061351, -0.0152516, -0.0062073, -0.0068810, 0.0071178
1: -0.0072125, -0.0046684, -0.0072387, -0.0046887, -0.0019400, 0.0020068
2: -0.0146556, 0.0041156, -0.0148487, 0.0039655, -0.0143139, 0.0148065
3: -0.0003121, 0.0021719, -0.0003377, 0.0021521, -0.0018942, 0.0019594
4: 0.0030161, 0.0170445, 0.0031283, 0.0171888, -0.0110654, 0.0106973
5: 0.9963443, 1.0002418, 0.9963754, 1.0002818, -0.0030743, 0.0029720
6: 0.0045653, 0.0081031, 0.0045936, 0.0081394, -0.0027905, 0.0026977
7: -0.0063446, 0.0068577, -0.0062391, 0.0069935, -0.0104138, 0.0100673
8: -0.0145302, -0.0042548, -0.0146359, -0.0043370, -0.0078354, 0.0081051
9: -0.0036426, -0.0027561, -0.0036356, -0.0027470, -0.0006993, 0.0006760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0021286
time: 1.92 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021555, upper bound: 0.0021556
time: 1.83 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0152318, -0.0062112, -0.0151673, -0.0060016, -0.0073396, 0.0070042
1: -0.0072331, -0.0046898, -0.0072149, -0.0046307, -0.0020693, 0.0019747
2: -0.0148074, 0.0039573, -0.0146732, 0.0043933, -0.0152679, 0.0145701
3: -0.0003322, 0.0021510, -0.0003145, 0.0022087, -0.0020205, 0.0019281
4: 0.0031344, 0.0171580, 0.0028086, 0.0170577, -0.0108888, 0.0114103
5: 0.9963771, 1.0002732, 0.9962866, 1.0002453, -0.0030252, 0.0031701
6: 0.0045951, 0.0081317, 0.0045130, 0.0081064, -0.0027460, 0.0028775
7: -0.0062333, 0.0069644, -0.0065399, 0.0068700, -0.0102476, 0.0107384
8: -0.0146133, -0.0043415, -0.0145398, -0.0041028, -0.0083577, 0.0079757
9: -0.0036352, -0.0027490, -0.0036558, -0.0027553, -0.0006881, 0.0007211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021293, upper bound: 0.0020281
time: 1.32 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021552, upper bound: 0.0021536
time: 1.43 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0152516, -0.0062073, -0.0152332, -0.0058773, -0.0075049, 0.0070620
1: -0.0072387, -0.0046887, -0.0072335, -0.0045957, -0.0021159, 0.0019910
2: -0.0148487, 0.0039655, -0.0148104, 0.0046519, -0.0156117, 0.0146904
3: -0.0003377, 0.0021521, -0.0003326, 0.0022429, -0.0020660, 0.0019440
4: 0.0031283, 0.0171888, 0.0026153, 0.0171602, -0.0109787, 0.0116672
5: 0.9963754, 1.0002818, 0.9962329, 1.0002738, -0.0030502, 0.0032415
6: 0.0045936, 0.0081394, 0.0044642, 0.0081322, -0.0027687, 0.0029423
7: -0.0062391, 0.0069935, -0.0067218, 0.0069665, -0.0103321, 0.0109801
8: -0.0146359, -0.0043370, -0.0146149, -0.0039613, -0.0085459, 0.0080415
9: -0.0036356, -0.0027470, -0.0036680, -0.0027488, -0.0006938, 0.0007373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021521, upper bound: 0.0020281
time: 2.02 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021785, upper bound: 0.0021536
time: 1.45 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0151673, -0.0060016, -0.0152318, -0.0062112, -0.0070042, 0.0073396
1: -0.0072149, -0.0046307, -0.0072331, -0.0046898, -0.0019747, 0.0020693
2: -0.0146732, 0.0043933, -0.0148074, 0.0039573, -0.0145701, 0.0152679
3: -0.0003145, 0.0022087, -0.0003322, 0.0021510, -0.0019281, 0.0020205
4: 0.0028086, 0.0170577, 0.0031344, 0.0171580, -0.0114103, 0.0108888
5: 0.9962866, 1.0002453, 0.9963771, 1.0002732, -0.0031701, 0.0030252
6: 0.0045130, 0.0081064, 0.0045951, 0.0081317, -0.0028775, 0.0027460
7: -0.0065399, 0.0068700, -0.0062333, 0.0069644, -0.0107384, 0.0102476
8: -0.0145398, -0.0041028, -0.0146133, -0.0043415, -0.0079757, 0.0083577
9: -0.0036558, -0.0027553, -0.0036352, -0.0027490, -0.0007211, 0.0006881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021292
time: 1.86 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021536, upper bound: 0.0021552
time: 1.81 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0152332, -0.0058773, -0.0152516, -0.0062073, -0.0070620, 0.0075049
1: -0.0072335, -0.0045957, -0.0072387, -0.0046887, -0.0019910, 0.0021159
2: -0.0148104, 0.0046519, -0.0148487, 0.0039655, -0.0146904, 0.0156117
3: -0.0003326, 0.0022429, -0.0003377, 0.0021521, -0.0019440, 0.0020660
4: 0.0026153, 0.0171602, 0.0031283, 0.0171888, -0.0116672, 0.0109787
5: 0.9962329, 1.0002738, 0.9963754, 1.0002818, -0.0032415, 0.0030502
6: 0.0044642, 0.0081322, 0.0045936, 0.0081394, -0.0029423, 0.0027687
7: -0.0067218, 0.0069665, -0.0062391, 0.0069935, -0.0109801, 0.0103321
8: -0.0146149, -0.0039613, -0.0146359, -0.0043370, -0.0080415, 0.0085459
9: -0.0036680, -0.0027488, -0.0036356, -0.0027470, -0.0007373, 0.0006938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021521
time: 1.49 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021535, upper bound: 0.0021787
time: 1.51 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0151673, -0.0060016, -0.0153179, -0.0059422, -0.0070020, 0.0071125
1: -0.0072149, -0.0046307, -0.0072573, -0.0046140, -0.0019741, 0.0020053
2: -0.0146732, 0.0043933, -0.0149865, 0.0045169, -0.0145656, 0.0147955
3: -0.0003145, 0.0022087, -0.0003559, 0.0022250, -0.0019275, 0.0019579
4: 0.0028086, 0.0170577, 0.0027163, 0.0172918, -0.0110572, 0.0108855
5: 0.9962866, 1.0002453, 0.9962609, 1.0003104, -0.0030720, 0.0030243
6: 0.0045130, 0.0081064, 0.0044897, 0.0081654, -0.0027885, 0.0027452
7: -0.0065399, 0.0068700, -0.0066268, 0.0070904, -0.0104061, 0.0102444
8: -0.0145398, -0.0041028, -0.0147113, -0.0040352, -0.0079733, 0.0080991
9: -0.0036558, -0.0027553, -0.0036616, -0.0027405, -0.0006987, 0.0006879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021292
time: 1.90 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021533, upper bound: 0.0021552
time: 1.86 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0152332, -0.0058773, -0.0153316, -0.0059383, -0.0070562, 0.0072979
1: -0.0072335, -0.0045957, -0.0072612, -0.0046129, -0.0019894, 0.0020576
2: -0.0148104, 0.0046519, -0.0150150, 0.0045250, -0.0146783, 0.0151812
3: -0.0003326, 0.0022429, -0.0003597, 0.0022261, -0.0019424, 0.0020090
4: 0.0026153, 0.0171602, 0.0027102, 0.0173131, -0.0113455, 0.0109697
5: 0.9962329, 1.0002738, 0.9962592, 1.0003164, -0.0031521, 0.0030477
6: 0.0044642, 0.0081322, 0.0044881, 0.0081708, -0.0028612, 0.0027664
7: -0.0067218, 0.0069665, -0.0066326, 0.0071104, -0.0106774, 0.0103237
8: -0.0146149, -0.0039613, -0.0147269, -0.0040307, -0.0080349, 0.0083102
9: -0.0036680, -0.0027488, -0.0036620, -0.0027392, -0.0007170, 0.0006932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021522
time: 1.42 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021534, upper bound: 0.0021786
time: 1.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.70 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0021024
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 5, lower bound: -0.0021556, upper bound: 0.0021286
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0021286
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 5, lower bound: -0.0021555, upper bound: 0.0021556
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 5, lower bound: -0.0021293, upper bound: 0.0020281
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 5, lower bound: -0.0021552, upper bound: 0.0021536
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 5, lower bound: -0.0021521, upper bound: 0.0020281
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 5, lower bound: -0.0021785, upper bound: 0.0021536
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021292
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 5, lower bound: -0.0021536, upper bound: 0.0021552
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021521
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 5, lower bound: -0.0021535, upper bound: 0.0021787
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021292
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 5, lower bound: -0.0021533, upper bound: 0.0021552
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021522
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 5, lower bound: -0.0021534, upper bound: 0.0021786

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0149213, -0.0062961, -0.0146761, -0.0061490, -0.0068265, 0.0063934
1: -0.0071455, -0.0047138, -0.0070764, -0.0046723, -0.0019246, 0.0018026
2: -0.0141614, 0.0037807, -0.0136515, 0.0040866, -0.0142004, 0.0132997
3: -0.0002467, 0.0021276, -0.0001793, 0.0021681, -0.0018792, 0.0017600
4: 0.0032664, 0.0166752, 0.0030378, 0.0162941, -0.0099393, 0.0106125
5: 0.9964138, 1.0001391, 0.9963502, 1.0000333, -0.0027614, 0.0029485
6: 0.0046284, 0.0080099, 0.0045707, 0.0079138, -0.0025066, 0.0026763
7: -0.0061091, 0.0065101, -0.0063243, 0.0061515, -0.0093540, 0.0099876
8: -0.0142597, -0.0044381, -0.0139806, -0.0042707, -0.0077733, 0.0072803
9: -0.0036268, -0.0027795, -0.0036413, -0.0028036, -0.0006281, 0.0006706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 233

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019341, upper bound: 0.0020345
time: 1.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019261, upper bound: 0.0020246
time: 1.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0150751, -0.0062721, -0.0151616, -0.0062268, -0.0068579, 0.0067886
1: -0.0071889, -0.0047070, -0.0072133, -0.0046942, -0.0019335, 0.0019140
2: -0.0144813, 0.0038306, -0.0146613, 0.0039247, -0.0142659, 0.0141217
3: -0.0002891, 0.0021342, -0.0003129, 0.0021467, -0.0018879, 0.0018688
4: 0.0032291, 0.0169143, 0.0031588, 0.0170488, -0.0105537, 0.0106614
5: 0.9964034, 1.0002056, 0.9963838, 1.0002429, -0.0029321, 0.0029621
6: 0.0046190, 0.0080702, 0.0046013, 0.0081041, -0.0026615, 0.0026887
7: -0.0061442, 0.0067351, -0.0062104, 0.0068617, -0.0099322, 0.0100336
8: -0.0144348, -0.0044109, -0.0145333, -0.0043593, -0.0078092, 0.0077303
9: -0.0036292, -0.0027644, -0.0036336, -0.0027559, -0.0006669, 0.0006737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021286, upper bound: 0.0020142
time: 1.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021286, upper bound: 0.0021285
time: 2.00 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0150044, -0.0061592, -0.0146946, -0.0061453, -0.0068091, 0.0065738
1: -0.0071690, -0.0046752, -0.0070816, -0.0046712, -0.0019197, 0.0018534
2: -0.0143344, 0.0040654, -0.0136898, 0.0040944, -0.0141642, 0.0136749
3: -0.0002696, 0.0021653, -0.0001843, 0.0021691, -0.0018744, 0.0018097
4: 0.0030536, 0.0168045, 0.0030320, 0.0163228, -0.0102198, 0.0105854
5: 0.9963546, 1.0001750, 0.9963486, 1.0000412, -0.0028394, 0.0029410
6: 0.0045748, 0.0080425, 0.0045693, 0.0079210, -0.0025773, 0.0026695
7: -0.0063093, 0.0066318, -0.0063297, 0.0061784, -0.0096179, 0.0099621
8: -0.0143544, -0.0042823, -0.0140015, -0.0042665, -0.0077535, 0.0074857
9: -0.0036403, -0.0027713, -0.0036416, -0.0028018, -0.0006458, 0.0006689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 233

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019341, upper bound: 0.0020640
time: 1.81 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019261, upper bound: 0.0020544
time: 1.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0151588, -0.0061351, -0.0151813, -0.0062230, -0.0068526, 0.0069479
1: -0.0072125, -0.0046684, -0.0072188, -0.0046932, -0.0019320, 0.0019589
2: -0.0146556, 0.0041156, -0.0147023, 0.0039327, -0.0142548, 0.0144531
3: -0.0003121, 0.0021719, -0.0003183, 0.0021477, -0.0018864, 0.0019126
4: 0.0030161, 0.0170445, 0.0031528, 0.0170795, -0.0108013, 0.0106531
5: 0.9963443, 1.0002418, 0.9963822, 1.0002514, -0.0030009, 0.0029598
6: 0.0045653, 0.0081031, 0.0045998, 0.0081119, -0.0027239, 0.0026866
7: -0.0063446, 0.0068577, -0.0062160, 0.0068906, -0.0101653, 0.0100258
8: -0.0145302, -0.0042548, -0.0145558, -0.0043549, -0.0078031, 0.0079116
9: -0.0036426, -0.0027561, -0.0036340, -0.0027539, -0.0006826, 0.0006732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021287, upper bound: 0.0020299
time: 1.37 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021287, upper bound: 0.0021556
time: 1.89 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0146761, -0.0061490, -0.0150178, -0.0060266, -0.0067994, 0.0069505
1: -0.0070764, -0.0046723, -0.0071727, -0.0046378, -0.0019170, 0.0019596
2: -0.0136515, 0.0040866, -0.0143622, 0.0043414, -0.0141441, 0.0144584
3: -0.0001793, 0.0021681, -0.0002733, 0.0022018, -0.0018718, 0.0019133
4: 0.0030378, 0.0162941, 0.0028474, 0.0168253, -0.0108053, 0.0105705
5: 0.9963502, 1.0000333, 0.9962974, 1.0001808, -0.0030020, 0.0029368
6: 0.0045707, 0.0079138, 0.0045227, 0.0080478, -0.0027249, 0.0026657
7: -0.0063243, 0.0061515, -0.0065034, 0.0066513, -0.0101690, 0.0099480
8: -0.0139806, -0.0042707, -0.0143696, -0.0041313, -0.0077425, 0.0079145
9: -0.0036413, -0.0028036, -0.0036533, -0.0027700, -0.0006828, 0.0006680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020580, upper bound: 0.0019310
time: 2.15 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020497, upper bound: 0.0019229
time: 1.95 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0151616, -0.0062268, -0.0151673, -0.0060016, -0.0072427, 0.0069757
1: -0.0072133, -0.0046942, -0.0072149, -0.0046307, -0.0020420, 0.0019667
2: -0.0146613, 0.0039247, -0.0146732, 0.0043933, -0.0150662, 0.0145109
3: -0.0003129, 0.0021467, -0.0003145, 0.0022087, -0.0019938, 0.0019203
4: 0.0031588, 0.0170488, 0.0028086, 0.0170577, -0.0108446, 0.0112596
5: 0.9963838, 1.0002429, 0.9962866, 1.0002453, -0.0030129, 0.0031282
6: 0.0046013, 0.0081041, 0.0045130, 0.0081064, -0.0027348, 0.0028395
7: -0.0062104, 0.0068617, -0.0065399, 0.0068700, -0.0102059, 0.0105965
8: -0.0145333, -0.0043593, -0.0145398, -0.0041028, -0.0082473, 0.0079433
9: -0.0036336, -0.0027559, -0.0036558, -0.0027553, -0.0006853, 0.0007115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020461, upper bound: 0.0021272
time: 1.87 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020461, upper bound: 0.0021536
time: 2.00 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0146946, -0.0061453, -0.0150822, -0.0059019, -0.0069598, 0.0070080
1: -0.0070816, -0.0046712, -0.0071909, -0.0046026, -0.0019622, 0.0019758
2: -0.0136898, 0.0040944, -0.0144962, 0.0046006, -0.0144777, 0.0145781
3: -0.0001843, 0.0021691, -0.0002910, 0.0022361, -0.0019159, 0.0019292
4: 0.0030320, 0.0163228, 0.0026537, 0.0169254, -0.0108947, 0.0108197
5: 0.9963486, 1.0000412, 0.9962435, 1.0002086, -0.0030269, 0.0030060
6: 0.0045693, 0.0079210, 0.0044739, 0.0080730, -0.0027475, 0.0027286
7: -0.0063297, 0.0061784, -0.0066858, 0.0067456, -0.0102532, 0.0101826
8: -0.0140015, -0.0042665, -0.0144430, -0.0039893, -0.0079251, 0.0079800
9: -0.0036416, -0.0028018, -0.0036656, -0.0027637, -0.0006885, 0.0006837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020826, upper bound: 0.0019310
time: 1.85 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020730, upper bound: 0.0019229
time: 1.85 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0151813, -0.0062230, -0.0152332, -0.0058773, -0.0073820, 0.0070336
1: -0.0072188, -0.0046932, -0.0072335, -0.0045957, -0.0020813, 0.0019830
2: -0.0147023, 0.0039327, -0.0148104, 0.0046519, -0.0153560, 0.0146313
3: -0.0003183, 0.0021477, -0.0003326, 0.0022429, -0.0020321, 0.0019362
4: 0.0031528, 0.0170795, 0.0026153, 0.0171602, -0.0109345, 0.0114761
5: 0.9963822, 1.0002514, 0.9962329, 1.0002738, -0.0030379, 0.0031884
6: 0.0045998, 0.0081119, 0.0044642, 0.0081322, -0.0027575, 0.0028941
7: -0.0062160, 0.0068906, -0.0067218, 0.0069665, -0.0102906, 0.0108003
8: -0.0145558, -0.0043549, -0.0146149, -0.0039613, -0.0084059, 0.0080092
9: -0.0036340, -0.0027539, -0.0036680, -0.0027488, -0.0006910, 0.0007252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020576, upper bound: 0.0021271
time: 1.78 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020576, upper bound: 0.0021536
time: 1.96 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0150178, -0.0060266, -0.0146761, -0.0061490, -0.0069505, 0.0067994
1: -0.0071727, -0.0046378, -0.0070764, -0.0046723, -0.0019596, 0.0019170
2: -0.0143622, 0.0043414, -0.0136515, 0.0040866, -0.0144584, 0.0141441
3: -0.0002733, 0.0022018, -0.0001793, 0.0021681, -0.0019133, 0.0018718
4: 0.0028474, 0.0168253, 0.0030378, 0.0162941, -0.0105705, 0.0108053
5: 0.9962974, 1.0001808, 0.9963502, 1.0000333, -0.0029368, 0.0030020
6: 0.0045227, 0.0080478, 0.0045707, 0.0079138, -0.0026657, 0.0027249
7: -0.0065034, 0.0066513, -0.0063243, 0.0061515, -0.0099480, 0.0101690
8: -0.0143696, -0.0041313, -0.0139806, -0.0042707, -0.0079145, 0.0077425
9: -0.0036533, -0.0027700, -0.0036413, -0.0028036, -0.0006680, 0.0006828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 233

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019309, upper bound: 0.0020580
time: 1.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0020495
time: 1.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0151673, -0.0060016, -0.0151616, -0.0062268, -0.0069757, 0.0072427
1: -0.0072149, -0.0046307, -0.0072133, -0.0046942, -0.0019667, 0.0020420
2: -0.0146732, 0.0043933, -0.0146613, 0.0039247, -0.0145109, 0.0150662
3: -0.0003145, 0.0022087, -0.0003129, 0.0021467, -0.0019203, 0.0019938
4: 0.0028086, 0.0170577, 0.0031588, 0.0170488, -0.0112596, 0.0108446
5: 0.9962866, 1.0002453, 0.9963838, 1.0002429, -0.0031282, 0.0030129
6: 0.0045130, 0.0081064, 0.0046013, 0.0081041, -0.0028395, 0.0027348
7: -0.0065399, 0.0068700, -0.0062104, 0.0068617, -0.0105965, 0.0102059
8: -0.0145398, -0.0041028, -0.0145333, -0.0043593, -0.0079433, 0.0082473
9: -0.0036558, -0.0027553, -0.0036336, -0.0027559, -0.0007115, 0.0006853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021271, upper bound: 0.0020461
time: 1.41 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021271, upper bound: 0.0021552
time: 2.00 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0150822, -0.0059019, -0.0146946, -0.0061453, -0.0070080, 0.0069598
1: -0.0071909, -0.0046026, -0.0070816, -0.0046712, -0.0019758, 0.0019622
2: -0.0144962, 0.0046006, -0.0136898, 0.0040944, -0.0145781, 0.0144777
3: -0.0002910, 0.0022361, -0.0001843, 0.0021691, -0.0019292, 0.0019159
4: 0.0026537, 0.0169254, 0.0030320, 0.0163228, -0.0108197, 0.0108947
5: 0.9962435, 1.0002086, 0.9963486, 1.0000412, -0.0030060, 0.0030269
6: 0.0044739, 0.0080730, 0.0045693, 0.0079210, -0.0027286, 0.0027475
7: -0.0066858, 0.0067456, -0.0063297, 0.0061784, -0.0101826, 0.0102532
8: -0.0144430, -0.0039893, -0.0140015, -0.0042665, -0.0079800, 0.0079251
9: -0.0036656, -0.0027637, -0.0036416, -0.0028018, -0.0006837, 0.0006885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 233

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019309, upper bound: 0.0020826
time: 1.88 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0020729
time: 1.84 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0152332, -0.0058773, -0.0151813, -0.0062230, -0.0070336, 0.0073820
1: -0.0072335, -0.0045957, -0.0072188, -0.0046932, -0.0019830, 0.0020813
2: -0.0148104, 0.0046519, -0.0147023, 0.0039327, -0.0146313, 0.0153560
3: -0.0003326, 0.0022429, -0.0003183, 0.0021477, -0.0019362, 0.0020321
4: 0.0026153, 0.0171602, 0.0031528, 0.0170795, -0.0114761, 0.0109345
5: 0.9962329, 1.0002738, 0.9963822, 1.0002514, -0.0031884, 0.0030379
6: 0.0044642, 0.0081322, 0.0045998, 0.0081119, -0.0028941, 0.0027575
7: -0.0067218, 0.0069665, -0.0062160, 0.0068906, -0.0108003, 0.0102906
8: -0.0146149, -0.0039613, -0.0145558, -0.0043549, -0.0080092, 0.0084059
9: -0.0036680, -0.0027488, -0.0036340, -0.0027539, -0.0007252, 0.0006910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021271, upper bound: 0.0020576
time: 2.06 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021271, upper bound: 0.0021786
time: 2.09 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0150178, -0.0060266, -0.0147800, -0.0058700, -0.0069440, 0.0065729
1: -0.0071727, -0.0046378, -0.0071057, -0.0045936, -0.0019578, 0.0018532
2: -0.0143622, 0.0043414, -0.0138675, 0.0046671, -0.0144450, 0.0136730
3: -0.0002733, 0.0022018, -0.0002079, 0.0022449, -0.0019116, 0.0018094
4: 0.0028474, 0.0168253, 0.0026040, 0.0164556, -0.0102184, 0.0107953
5: 0.9962974, 1.0001808, 0.9962296, 1.0000781, -0.0028390, 0.0029992
6: 0.0045227, 0.0080478, 0.0044614, 0.0079545, -0.0025769, 0.0027224
7: -0.0065034, 0.0066513, -0.0067325, 0.0063034, -0.0096166, 0.0101595
8: -0.0143696, -0.0041313, -0.0140988, -0.0039529, -0.0079072, 0.0074846
9: -0.0036533, -0.0027700, -0.0036687, -0.0027934, -0.0006457, 0.0006822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 233

## Relational analysis of IS_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019310, upper bound: 0.0020580
time: 1.53 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0020497
time: 1.38 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0151673, -0.0060016, -0.0152442, -0.0059578, -0.0069735, 0.0069912
1: -0.0072149, -0.0046307, -0.0072366, -0.0046184, -0.0019661, 0.0019711
2: -0.0146732, 0.0043933, -0.0148332, 0.0044844, -0.0145062, 0.0145431
3: -0.0003145, 0.0022087, -0.0003356, 0.0022207, -0.0019197, 0.0019246
4: 0.0028086, 0.0170577, 0.0027405, 0.0171773, -0.0108686, 0.0108411
5: 0.9962866, 1.0002453, 0.9962677, 1.0002786, -0.0030196, 0.0030120
6: 0.0045130, 0.0081064, 0.0044958, 0.0081365, -0.0027409, 0.0027340
7: -0.0065399, 0.0068700, -0.0066040, 0.0069826, -0.0102286, 0.0102026
8: -0.0145398, -0.0041028, -0.0146274, -0.0040529, -0.0079407, 0.0079609
9: -0.0036558, -0.0027553, -0.0036601, -0.0027478, -0.0006868, 0.0006851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021270, upper bound: 0.0020461
time: 1.97 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021270, upper bound: 0.0021552
time: 1.85 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0150822, -0.0059019, -0.0147938, -0.0058665, -0.0069982, 0.0067559
1: -0.0071909, -0.0046026, -0.0071096, -0.0045927, -0.0019730, 0.0019048
2: -0.0144962, 0.0046006, -0.0138962, 0.0046743, -0.0145576, 0.0140537
3: -0.0002910, 0.0022361, -0.0002116, 0.0022459, -0.0019265, 0.0018598
4: 0.0026537, 0.0169254, 0.0025986, 0.0164770, -0.0105029, 0.0108795
5: 0.9962435, 1.0002086, 0.9962282, 1.0000840, -0.0029180, 0.0030226
6: 0.0044739, 0.0080730, 0.0044600, 0.0079599, -0.0026487, 0.0027436
7: -0.0066858, 0.0067456, -0.0067375, 0.0063236, -0.0098844, 0.0102388
8: -0.0144430, -0.0039893, -0.0141145, -0.0039490, -0.0079689, 0.0076930
9: -0.0036656, -0.0027637, -0.0036690, -0.0027920, -0.0006637, 0.0006875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 233

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019310, upper bound: 0.0020827
time: 2.19 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0020729
time: 1.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0152332, -0.0058773, -0.0152592, -0.0059540, -0.0070277, 0.0071395
1: -0.0072335, -0.0045957, -0.0072408, -0.0046173, -0.0019814, 0.0020129
2: -0.0148104, 0.0046519, -0.0148645, 0.0044924, -0.0146190, 0.0148516
3: -0.0003326, 0.0022429, -0.0003398, 0.0022218, -0.0019346, 0.0019654
4: 0.0026153, 0.0171602, 0.0027346, 0.0172006, -0.0110991, 0.0109253
5: 0.9962329, 1.0002738, 0.9962659, 1.0002850, -0.0030837, 0.0030354
6: 0.0044642, 0.0081322, 0.0044943, 0.0081424, -0.0027990, 0.0027552
7: -0.0067218, 0.0069665, -0.0066096, 0.0070046, -0.0104455, 0.0102820
8: -0.0146149, -0.0039613, -0.0146445, -0.0040486, -0.0080025, 0.0081298
9: -0.0036680, -0.0027488, -0.0036604, -0.0027463, -0.0007014, 0.0006904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021270, upper bound: 0.0020576
time: 1.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021270, upper bound: 0.0021786
time: 2.12 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.99 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0019341, upper bound: 0.0020345
IS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0019261, upper bound: 0.0020246
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0021286, upper bound: 0.0020142
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0021286, upper bound: 0.0021285
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0019341, upper bound: 0.0020640
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0019261, upper bound: 0.0020544
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0021287, upper bound: 0.0020299
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0021287, upper bound: 0.0021556
IS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0020580, upper bound: 0.0019310
IS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0020497, upper bound: 0.0019229
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0020461, upper bound: 0.0021272
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0020461, upper bound: 0.0021536
IS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0020826, upper bound: 0.0019310
IS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0020730, upper bound: 0.0019229
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0020576, upper bound: 0.0021271
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0020576, upper bound: 0.0021536
IS_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0019309, upper bound: 0.0020580
IS_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0020495
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0021271, upper bound: 0.0020461
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0021271, upper bound: 0.0021552
IS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0019309, upper bound: 0.0020826
IS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0020729
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0021271, upper bound: 0.0020576
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0021271, upper bound: 0.0021786
IS_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0019310, upper bound: 0.0020580
IS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0020497
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0021270, upper bound: 0.0020461
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0021270, upper bound: 0.0021552
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0019310, upper bound: 0.0020827
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0020729
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0021270, upper bound: 0.0020576
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 5, lower bound: -0.0021270, upper bound: 0.0021786

## BFS IS instance: IS_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0149213, -0.0062961, -0.0145763, -0.0061975, -0.0067806, 0.0062858
1: -0.0071455, -0.0047138, -0.0070483, -0.0046860, -0.0019117, 0.0017722
2: -0.0141614, 0.0037807, -0.0134438, 0.0039858, -0.0141050, 0.0130758
3: -0.0002467, 0.0021276, -0.0001518, 0.0021547, -0.0018666, 0.0017304
4: 0.0032664, 0.0166752, 0.0031131, 0.0161389, -0.0097720, 0.0105412
5: 0.9964138, 1.0001391, 0.9963712, 0.9999902, -0.0027150, 0.0029287
6: 0.0046284, 0.0080099, 0.0045898, 0.0078747, -0.0024644, 0.0026583
7: -0.0061091, 0.0065101, -0.0062533, 0.0060054, -0.0091966, 0.0099205
8: -0.0142597, -0.0044381, -0.0138669, -0.0043259, -0.0077211, 0.0071577
9: -0.0036268, -0.0027795, -0.0036365, -0.0028134, -0.0006175, 0.0006661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018949, upper bound: 0.0020076
time: 1.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019063, upper bound: 0.0020033
time: 1.49 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0148669, -0.0063243, -0.0145206, -0.0059505, -0.0069766, 0.0063154
1: -0.0071302, -0.0047217, -0.0070325, -0.0046163, -0.0019670, 0.0017805
2: -0.0140483, 0.0037220, -0.0133279, 0.0044997, -0.0145127, 0.0131372
3: -0.0002318, 0.0021198, -0.0001364, 0.0022228, -0.0019205, 0.0017385
4: 0.0033103, 0.0165907, 0.0027291, 0.0160523, -0.0098179, 0.0108459
5: 0.9964259, 1.0001156, 0.9962645, 0.9999660, -0.0027277, 0.0030133
6: 0.0046395, 0.0079886, 0.0044929, 0.0078528, -0.0024759, 0.0027352
7: -0.0060678, 0.0064305, -0.0066148, 0.0059238, -0.0092398, 0.0102072
8: -0.0141978, -0.0044703, -0.0138034, -0.0040446, -0.0079443, 0.0071913
9: -0.0036241, -0.0027848, -0.0036608, -0.0028188, -0.0006204, 0.0006854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018860, upper bound: 0.0019945
time: 1.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018988, upper bound: 0.0019940
time: 1.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0145236, -0.0062049, -0.0151616, -0.0062268, -0.0063714, 0.0069539
1: -0.0070334, -0.0046881, -0.0072133, -0.0046942, -0.0017963, 0.0019606
2: -0.0133342, 0.0039704, -0.0146613, 0.0039247, -0.0132539, 0.0144655
3: -0.0001373, 0.0021527, -0.0003129, 0.0021467, -0.0017539, 0.0019143
4: 0.0031246, 0.0160570, 0.0031588, 0.0170488, -0.0108106, 0.0099051
5: 0.9963744, 0.9999673, 0.9963838, 1.0002429, -0.0030035, 0.0027519
6: 0.0045927, 0.0078540, 0.0046013, 0.0081041, -0.0027263, 0.0024979
7: -0.0062425, 0.0059283, -0.0062104, 0.0068617, -0.0101740, 0.0093218
8: -0.0138069, -0.0043343, -0.0145333, -0.0043593, -0.0072552, 0.0079184
9: -0.0036358, -0.0028185, -0.0036336, -0.0027559, -0.0006832, 0.0006259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of IS_A1_B1_A1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019388, upper bound: 0.0019183
time: 1.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019261, upper bound: 0.0019104
time: 1.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0150048, -0.0062870, -0.0151616, -0.0062268, -0.0067654, 0.0067632
1: -0.0071691, -0.0047112, -0.0072133, -0.0046942, -0.0019074, 0.0019068
2: -0.0143353, 0.0037996, -0.0146613, 0.0039247, -0.0140734, 0.0140689
3: -0.0002697, 0.0021301, -0.0003129, 0.0021467, -0.0018624, 0.0018618
4: 0.0032522, 0.0168052, 0.0031588, 0.0170488, -0.0105142, 0.0105176
5: 0.9964098, 1.0001752, 0.9963838, 1.0002429, -0.0029212, 0.0029221
6: 0.0046248, 0.0080427, 0.0046013, 0.0081041, -0.0026515, 0.0026524
7: -0.0061224, 0.0066324, -0.0062104, 0.0068617, -0.0098950, 0.0098982
8: -0.0143549, -0.0044278, -0.0145333, -0.0043593, -0.0077038, 0.0077013
9: -0.0036277, -0.0027713, -0.0036336, -0.0027559, -0.0006644, 0.0006646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019993, upper bound: 0.0020839
time: 1.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019993, upper bound: 0.0021025
time: 2.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0150044, -0.0061592, -0.0145949, -0.0061940, -0.0067625, 0.0064637
1: -0.0071690, -0.0046752, -0.0070535, -0.0046850, -0.0019066, 0.0018224
2: -0.0143344, 0.0040654, -0.0134825, 0.0039931, -0.0140673, 0.0134458
3: -0.0002696, 0.0021653, -0.0001569, 0.0021557, -0.0018616, 0.0017793
4: 0.0030536, 0.0168045, 0.0031077, 0.0161679, -0.0100485, 0.0105130
5: 0.9963546, 1.0001750, 0.9963697, 0.9999982, -0.0027918, 0.0029208
6: 0.0045748, 0.0080425, 0.0045884, 0.0078820, -0.0025341, 0.0026512
7: -0.0063093, 0.0066318, -0.0062585, 0.0060326, -0.0094568, 0.0098939
8: -0.0143544, -0.0042823, -0.0138881, -0.0043219, -0.0077005, 0.0073602
9: -0.0036403, -0.0027713, -0.0036369, -0.0028115, -0.0006350, 0.0006644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018949, upper bound: 0.0020366
time: 1.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019063, upper bound: 0.0020333
time: 1.97 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0149486, -0.0061885, -0.0145386, -0.0059473, -0.0070266, 0.0064903
1: -0.0071532, -0.0046834, -0.0070376, -0.0046154, -0.0019811, 0.0018298
2: -0.0142182, 0.0040044, -0.0133654, 0.0045062, -0.0146169, 0.0135011
3: -0.0002543, 0.0021572, -0.0001414, 0.0022236, -0.0019343, 0.0017867
4: 0.0030992, 0.0167177, 0.0027242, 0.0160803, -0.0100899, 0.0109237
5: 0.9963673, 1.0001509, 0.9962630, 0.9999738, -0.0028033, 0.0030349
6: 0.0045862, 0.0080206, 0.0044917, 0.0078599, -0.0025445, 0.0027548
7: -0.0062665, 0.0065500, -0.0066193, 0.0059502, -0.0094957, 0.0102804
8: -0.0142908, -0.0043157, -0.0138239, -0.0040410, -0.0080013, 0.0073905
9: -0.0036374, -0.0027768, -0.0036611, -0.0028171, -0.0006376, 0.0006903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018860, upper bound: 0.0020230
time: 1.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018988, upper bound: 0.0020245
time: 1.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0146018, -0.0060723, -0.0151813, -0.0062230, -0.0063571, 0.0071150
1: -0.0070554, -0.0046507, -0.0072188, -0.0046932, -0.0017923, 0.0020060
2: -0.0134968, 0.0042463, -0.0147023, 0.0039327, -0.0132241, 0.0148006
3: -0.0001588, 0.0021892, -0.0003183, 0.0021477, -0.0017500, 0.0019586
4: 0.0029184, 0.0161786, 0.0031528, 0.0170795, -0.0110610, 0.0098829
5: 0.9963171, 1.0000011, 0.9963822, 1.0002514, -0.0030731, 0.0027458
6: 0.0045407, 0.0078847, 0.0045998, 0.0081119, -0.0027894, 0.0024923
7: -0.0064366, 0.0060427, -0.0062160, 0.0068906, -0.0104096, 0.0093009
8: -0.0138959, -0.0041833, -0.0145558, -0.0043549, -0.0072389, 0.0081018
9: -0.0036488, -0.0028109, -0.0036340, -0.0027539, -0.0006990, 0.0006245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of IS_A1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019388, upper bound: 0.0019339
time: 1.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019261, upper bound: 0.0019259
time: 1.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0150879, -0.0061514, -0.0151813, -0.0062230, -0.0067195, 0.0069204
1: -0.0071925, -0.0046730, -0.0072188, -0.0046932, -0.0018945, 0.0019511
2: -0.0145080, 0.0040818, -0.0147023, 0.0039327, -0.0139780, 0.0143959
3: -0.0002926, 0.0021675, -0.0003183, 0.0021477, -0.0018498, 0.0019051
4: 0.0030414, 0.0169342, 0.0031528, 0.0170795, -0.0107586, 0.0104463
5: 0.9963512, 1.0002111, 0.9963822, 1.0002514, -0.0029891, 0.0029023
6: 0.0045717, 0.0080752, 0.0045998, 0.0081119, -0.0027132, 0.0026344
7: -0.0063208, 0.0067538, -0.0062160, 0.0068906, -0.0101250, 0.0098311
8: -0.0144494, -0.0042734, -0.0145558, -0.0043549, -0.0076516, 0.0078803
9: -0.0036411, -0.0027631, -0.0036340, -0.0027539, -0.0006799, 0.0006601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019993, upper bound: 0.0021042
time: 1.94 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019992, upper bound: 0.0021324
time: 1.92 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0145763, -0.0061975, -0.0150178, -0.0060266, -0.0066918, 0.0069040
1: -0.0070483, -0.0046860, -0.0071727, -0.0046378, -0.0018867, 0.0019465
2: -0.0134438, 0.0039858, -0.0143622, 0.0043414, -0.0139203, 0.0143618
3: -0.0001518, 0.0021547, -0.0002733, 0.0022018, -0.0018421, 0.0019006
4: 0.0031131, 0.0161389, 0.0028474, 0.0168253, -0.0107331, 0.0104032
5: 0.9963712, 0.9999902, 0.9962974, 1.0001808, -0.0029820, 0.0028903
6: 0.0045898, 0.0078747, 0.0045227, 0.0080478, -0.0027067, 0.0026235
7: -0.0062533, 0.0060054, -0.0065034, 0.0066513, -0.0101010, 0.0097905
8: -0.0138669, -0.0043259, -0.0143696, -0.0041313, -0.0076200, 0.0078617
9: -0.0036365, -0.0028134, -0.0036533, -0.0027700, -0.0006783, 0.0006574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020279, upper bound: 0.0018904
time: 1.57 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020263, upper bound: 0.0019033
time: 1.84 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0145206, -0.0059505, -0.0149589, -0.0060544, -0.0067228, 0.0071544
1: -0.0070325, -0.0046163, -0.0071561, -0.0046456, -0.0018954, 0.0020171
2: -0.0133279, 0.0044997, -0.0142398, 0.0042835, -0.0139849, 0.0148825
3: -0.0001364, 0.0022228, -0.0002571, 0.0021941, -0.0018507, 0.0019695
4: 0.0027291, 0.0160523, 0.0028907, 0.0167338, -0.0111223, 0.0104514
5: 0.9962645, 0.9999660, 0.9963094, 1.0001553, -0.0030901, 0.0029037
6: 0.0044929, 0.0078528, 0.0045337, 0.0080247, -0.0028049, 0.0026357
7: -0.0066148, 0.0059238, -0.0064627, 0.0065652, -0.0104673, 0.0098360
8: -0.0138034, -0.0040446, -0.0143026, -0.0041629, -0.0076554, 0.0081467
9: -0.0036608, -0.0028188, -0.0036506, -0.0027758, -0.0007029, 0.0006605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020153, upper bound: 0.0018816
time: 1.52 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0018958
time: 1.87 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0151616, -0.0062268, -0.0146279, -0.0059244, -0.0074037, 0.0065333
1: -0.0072133, -0.0046942, -0.0070628, -0.0046090, -0.0020874, 0.0018420
2: -0.0146613, 0.0039247, -0.0135512, 0.0045538, -0.0154011, 0.0135906
3: -0.0003129, 0.0021467, -0.0001660, 0.0022299, -0.0020381, 0.0017985
4: 0.0031588, 0.0170488, 0.0026886, 0.0162192, -0.0101568, 0.0115098
5: 0.9963838, 1.0002429, 0.9962532, 1.0000124, -0.0028219, 0.0031978
6: 0.0046013, 0.0081041, 0.0044827, 0.0078949, -0.0025614, 0.0029026
7: -0.0062104, 0.0068617, -0.0066529, 0.0060810, -0.0095587, 0.0108320
8: -0.0145333, -0.0043593, -0.0139257, -0.0040149, -0.0084306, 0.0074395
9: -0.0036336, -0.0027559, -0.0036633, -0.0028083, -0.0006418, 0.0007274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 233

## Relational analysis of IS_A1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019465, upper bound: 0.0020608
time: 1.48 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019345, upper bound: 0.0020512
time: 1.64 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0151616, -0.0062268, -0.0150919, -0.0060165, -0.0072169, 0.0068802
1: -0.0072133, -0.0046942, -0.0071936, -0.0046349, -0.0020347, 0.0019398
2: -0.0146613, 0.0039247, -0.0145165, 0.0043623, -0.0150126, 0.0143121
3: -0.0003129, 0.0021467, -0.0002937, 0.0022046, -0.0019867, 0.0018940
4: 0.0031588, 0.0170488, 0.0028318, 0.0169406, -0.0106960, 0.0112195
5: 0.9963838, 1.0002429, 0.9962931, 1.0002128, -0.0029717, 0.0031171
6: 0.0046013, 0.0081041, 0.0045188, 0.0080768, -0.0026974, 0.0028294
7: -0.0062104, 0.0068617, -0.0065182, 0.0067598, -0.0100661, 0.0105588
8: -0.0145333, -0.0043593, -0.0144540, -0.0041198, -0.0082179, 0.0078345
9: -0.0036336, -0.0027559, -0.0036543, -0.0027627, -0.0006759, 0.0007090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of IS_A1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019466, upper bound: 0.0020889
time: 1.93 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019345, upper bound: 0.0021040
time: 1.84 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0145949, -0.0061940, -0.0150822, -0.0059019, -0.0068496, 0.0069614
1: -0.0070535, -0.0046850, -0.0071909, -0.0046026, -0.0019312, 0.0019627
2: -0.0134825, 0.0039931, -0.0144962, 0.0046006, -0.0142486, 0.0144811
3: -0.0001569, 0.0021557, -0.0002910, 0.0022361, -0.0018856, 0.0019163
4: 0.0031077, 0.0161679, 0.0026537, 0.0169254, -0.0108223, 0.0106485
5: 0.9963697, 0.9999982, 0.9962435, 1.0002086, -0.0030068, 0.0029585
6: 0.0045884, 0.0078820, 0.0044739, 0.0080730, -0.0027292, 0.0026854
7: -0.0062585, 0.0060326, -0.0066858, 0.0067456, -0.0101850, 0.0100214
8: -0.0138881, -0.0043219, -0.0144430, -0.0039893, -0.0077997, 0.0079270
9: -0.0036369, -0.0028115, -0.0036656, -0.0027637, -0.0006839, 0.0006729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020525, upper bound: 0.0018904
time: 2.00 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020499, upper bound: 0.0019034
time: 1.63 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0145386, -0.0059473, -0.0150223, -0.0059306, -0.0068779, 0.0071394
1: -0.0070376, -0.0046154, -0.0071740, -0.0046107, -0.0019391, 0.0020129
2: -0.0133654, 0.0045062, -0.0143715, 0.0045409, -0.0143075, 0.0148513
3: -0.0001414, 0.0022236, -0.0002745, 0.0022282, -0.0018934, 0.0019653
4: 0.0027242, 0.0160803, 0.0026983, 0.0168322, -0.0110990, 0.0106925
5: 0.9962630, 0.9999738, 0.9962559, 1.0001827, -0.0030836, 0.0029707
6: 0.0044917, 0.0078599, 0.0044851, 0.0080495, -0.0027990, 0.0026965
7: -0.0066193, 0.0059502, -0.0066438, 0.0066579, -0.0104454, 0.0100629
8: -0.0138239, -0.0040410, -0.0143747, -0.0040220, -0.0078319, 0.0081296
9: -0.0036611, -0.0028171, -0.0036627, -0.0027696, -0.0007014, 0.0006757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020376, upper bound: 0.0018816
time: 1.90 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020418, upper bound: 0.0018959
time: 1.90 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0151813, -0.0062230, -0.0146912, -0.0058057, -0.0075514, 0.0065910
1: -0.0072188, -0.0046932, -0.0070806, -0.0045755, -0.0021290, 0.0018583
2: -0.0147023, 0.0039327, -0.0136827, 0.0048008, -0.0157084, 0.0137107
3: -0.0003183, 0.0021477, -0.0001834, 0.0022626, -0.0020788, 0.0018144
4: 0.0031528, 0.0170795, 0.0025041, 0.0163175, -0.0102465, 0.0117395
5: 0.9963822, 1.0002514, 0.9962019, 1.0000397, -0.0028468, 0.0032616
6: 0.0045998, 0.0081119, 0.0044362, 0.0079197, -0.0025840, 0.0029605
7: -0.0062160, 0.0068906, -0.0068265, 0.0061734, -0.0096431, 0.0110482
8: -0.0145558, -0.0043549, -0.0139977, -0.0038798, -0.0085988, 0.0075052
9: -0.0036340, -0.0027539, -0.0036750, -0.0028021, -0.0006475, 0.0007419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 233

## Relational analysis of IS_A1_B2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019585, upper bound: 0.0020609
time: 1.74 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019496, upper bound: 0.0020512
time: 1.70 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0151813, -0.0062230, -0.0151599, -0.0058933, -0.0073547, 0.0069392
1: -0.0072188, -0.0046932, -0.0072128, -0.0046002, -0.0020736, 0.0019564
2: -0.0147023, 0.0039327, -0.0146578, 0.0046185, -0.0152992, 0.0144350
3: -0.0003183, 0.0021477, -0.0003124, 0.0022385, -0.0020246, 0.0019102
4: 0.0031528, 0.0170795, 0.0026403, 0.0170462, -0.0107878, 0.0114337
5: 0.9963822, 1.0002514, 0.9962398, 1.0002421, -0.0029972, 0.0031766
6: 0.0045998, 0.0081119, 0.0044705, 0.0081035, -0.0027205, 0.0028834
7: -0.0062160, 0.0068906, -0.0066983, 0.0068592, -0.0101525, 0.0107604
8: -0.0145558, -0.0043549, -0.0145314, -0.0039795, -0.0083748, 0.0079017
9: -0.0036340, -0.0027539, -0.0036664, -0.0027560, -0.0006817, 0.0007225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 233

## Relational analysis of IS_A1_B2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019584, upper bound: 0.0021041
time: 1.82 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019496, upper bound: 0.0021041
time: 1.92 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0150178, -0.0060266, -0.0145763, -0.0061975, -0.0069040, 0.0066918
1: -0.0071727, -0.0046378, -0.0070483, -0.0046860, -0.0019465, 0.0018867
2: -0.0143622, 0.0043414, -0.0134438, 0.0039858, -0.0143618, 0.0139203
3: -0.0002733, 0.0022018, -0.0001518, 0.0021547, -0.0019006, 0.0018421
4: 0.0028474, 0.0168253, 0.0031131, 0.0161389, -0.0104032, 0.0107331
5: 0.9962974, 1.0001808, 0.9963712, 0.9999902, -0.0028903, 0.0029820
6: 0.0045227, 0.0080478, 0.0045898, 0.0078747, -0.0026235, 0.0027067
7: -0.0065034, 0.0066513, -0.0062533, 0.0060054, -0.0097905, 0.0101010
8: -0.0143696, -0.0041313, -0.0138669, -0.0043259, -0.0078617, 0.0076200
9: -0.0036533, -0.0027700, -0.0036365, -0.0028134, -0.0006574, 0.0006783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A2_B1_A1_B1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018905, upper bound: 0.0020278
time: 1.95 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019034, upper bound: 0.0020264
time: 2.11 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0149589, -0.0060544, -0.0145206, -0.0059505, -0.0071544, 0.0067228
1: -0.0071561, -0.0046456, -0.0070325, -0.0046163, -0.0020171, 0.0018954
2: -0.0142398, 0.0042835, -0.0133279, 0.0044997, -0.0148825, 0.0139849
3: -0.0002571, 0.0021941, -0.0001364, 0.0022228, -0.0019695, 0.0018507
4: 0.0028907, 0.0167338, 0.0027291, 0.0160523, -0.0104514, 0.0111223
5: 0.9963094, 1.0001553, 0.9962645, 0.9999660, -0.0029037, 0.0030901
6: 0.0045337, 0.0080247, 0.0044929, 0.0078528, -0.0026357, 0.0028049
7: -0.0064627, 0.0065652, -0.0066148, 0.0059238, -0.0098360, 0.0104673
8: -0.0143026, -0.0041629, -0.0138034, -0.0040446, -0.0081467, 0.0076554
9: -0.0036506, -0.0027758, -0.0036608, -0.0028188, -0.0006605, 0.0007029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A2_B1_A1_B1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018816, upper bound: 0.0020153
time: 1.96 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018959, upper bound: 0.0020182
time: 1.83 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0146279, -0.0059244, -0.0151616, -0.0062268, -0.0065333, 0.0074037
1: -0.0070628, -0.0046090, -0.0072133, -0.0046942, -0.0018420, 0.0020874
2: -0.0135512, 0.0045538, -0.0146613, 0.0039247, -0.0135906, 0.0154011
3: -0.0001660, 0.0022299, -0.0003129, 0.0021467, -0.0017985, 0.0020381
4: 0.0026886, 0.0162192, 0.0031588, 0.0170488, -0.0115098, 0.0101568
5: 0.9962532, 1.0000124, 0.9963838, 1.0002429, -0.0031978, 0.0028219
6: 0.0044827, 0.0078949, 0.0046013, 0.0081041, -0.0029026, 0.0025614
7: -0.0066529, 0.0060810, -0.0062104, 0.0068617, -0.0108320, 0.0095587
8: -0.0139257, -0.0040149, -0.0145333, -0.0043593, -0.0074395, 0.0084306
9: -0.0036633, -0.0028083, -0.0036336, -0.0027559, -0.0007274, 0.0006418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of IS_A2_B1_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019364, upper bound: 0.0019465
time: 1.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019230, upper bound: 0.0019344
time: 1.98 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0150919, -0.0060165, -0.0151616, -0.0062268, -0.0068802, 0.0072169
1: -0.0071936, -0.0046349, -0.0072133, -0.0046942, -0.0019398, 0.0020347
2: -0.0145165, 0.0043623, -0.0146613, 0.0039247, -0.0143121, 0.0150126
3: -0.0002937, 0.0022046, -0.0003129, 0.0021467, -0.0018940, 0.0019867
4: 0.0028318, 0.0169406, 0.0031588, 0.0170488, -0.0112195, 0.0106960
5: 0.9962931, 1.0002128, 0.9963838, 1.0002429, -0.0031171, 0.0029717
6: 0.0045188, 0.0080768, 0.0046013, 0.0081041, -0.0028294, 0.0026974
7: -0.0065182, 0.0067598, -0.0062104, 0.0068617, -0.0105588, 0.0100661
8: -0.0144540, -0.0041198, -0.0145333, -0.0043593, -0.0078345, 0.0082179
9: -0.0036543, -0.0027627, -0.0036336, -0.0027559, -0.0007090, 0.0006759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 233

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019310, upper bound: 0.0021037
time: 1.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019230, upper bound: 0.0021038
time: 2.08 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0150822, -0.0059019, -0.0145949, -0.0061940, -0.0069614, 0.0068496
1: -0.0071909, -0.0046026, -0.0070535, -0.0046850, -0.0019627, 0.0019312
2: -0.0144962, 0.0046006, -0.0134825, 0.0039931, -0.0144811, 0.0142486
3: -0.0002910, 0.0022361, -0.0001569, 0.0021557, -0.0019163, 0.0018856
4: 0.0026537, 0.0169254, 0.0031077, 0.0161679, -0.0106485, 0.0108223
5: 0.9962435, 1.0002086, 0.9963697, 0.9999982, -0.0029585, 0.0030068
6: 0.0044739, 0.0080730, 0.0045884, 0.0078820, -0.0026854, 0.0027292
7: -0.0066858, 0.0067456, -0.0062585, 0.0060326, -0.0100214, 0.0101850
8: -0.0144430, -0.0039893, -0.0138881, -0.0043219, -0.0079270, 0.0077997
9: -0.0036656, -0.0027637, -0.0036369, -0.0028115, -0.0006729, 0.0006839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A2_B1_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018904, upper bound: 0.0020524
time: 2.05 seconds

## Relational analysis of IS_A2_B1_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019034, upper bound: 0.0020499
time: 1.85 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0150223, -0.0059306, -0.0145386, -0.0059473, -0.0071394, 0.0068779
1: -0.0071740, -0.0046107, -0.0070376, -0.0046154, -0.0020129, 0.0019391
2: -0.0143715, 0.0045409, -0.0133654, 0.0045062, -0.0148513, 0.0143075
3: -0.0002745, 0.0022282, -0.0001414, 0.0022236, -0.0019653, 0.0018934
4: 0.0026983, 0.0168322, 0.0027242, 0.0160803, -0.0106925, 0.0110990
5: 0.9962559, 1.0001827, 0.9962630, 0.9999738, -0.0029707, 0.0030836
6: 0.0044851, 0.0080495, 0.0044917, 0.0078599, -0.0026965, 0.0027990
7: -0.0066438, 0.0066579, -0.0066193, 0.0059502, -0.0100629, 0.0104454
8: -0.0143747, -0.0040220, -0.0138239, -0.0040410, -0.0081296, 0.0078319
9: -0.0036627, -0.0027696, -0.0036611, -0.0028171, -0.0006757, 0.0007014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A2_B1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018816, upper bound: 0.0020377
time: 1.84 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018959, upper bound: 0.0020418
time: 1.98 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0146912, -0.0058057, -0.0151813, -0.0062230, -0.0065910, 0.0075514
1: -0.0070806, -0.0045755, -0.0072188, -0.0046932, -0.0018583, 0.0021290
2: -0.0136827, 0.0048008, -0.0147023, 0.0039327, -0.0137107, 0.0157084
3: -0.0001834, 0.0022626, -0.0003183, 0.0021477, -0.0018144, 0.0020788
4: 0.0025041, 0.0163175, 0.0031528, 0.0170795, -0.0117395, 0.0102465
5: 0.9962019, 1.0000397, 0.9963822, 1.0002514, -0.0032616, 0.0028468
6: 0.0044362, 0.0079197, 0.0045998, 0.0081119, -0.0029605, 0.0025840
7: -0.0068265, 0.0061734, -0.0062160, 0.0068906, -0.0110482, 0.0096431
8: -0.0139977, -0.0038798, -0.0145558, -0.0043549, -0.0075052, 0.0085988
9: -0.0036750, -0.0028021, -0.0036340, -0.0027539, -0.0007419, 0.0006475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of IS_A2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019364, upper bound: 0.0019584
time: 1.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0019497
time: 1.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0151599, -0.0058933, -0.0151813, -0.0062230, -0.0069392, 0.0073547
1: -0.0072128, -0.0046002, -0.0072188, -0.0046932, -0.0019564, 0.0020736
2: -0.0146578, 0.0046185, -0.0147023, 0.0039327, -0.0144350, 0.0152992
3: -0.0003124, 0.0022385, -0.0003183, 0.0021477, -0.0019102, 0.0020246
4: 0.0026403, 0.0170462, 0.0031528, 0.0170795, -0.0114337, 0.0107878
5: 0.9962398, 1.0002421, 0.9963822, 1.0002514, -0.0031766, 0.0029972
6: 0.0044705, 0.0081035, 0.0045998, 0.0081119, -0.0028834, 0.0027205
7: -0.0066983, 0.0068592, -0.0062160, 0.0068906, -0.0107604, 0.0101525
8: -0.0145314, -0.0039795, -0.0145558, -0.0043549, -0.0079017, 0.0083748
9: -0.0036664, -0.0027560, -0.0036340, -0.0027539, -0.0007225, 0.0006817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of IS_A2_B1_A2_B2_A2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019364, upper bound: 0.0021179
time: 1.94 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0021274
time: 2.02 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0150178, -0.0060266, -0.0146808, -0.0059178, -0.0068978, 0.0064647
1: -0.0071727, -0.0046378, -0.0070777, -0.0046071, -0.0019448, 0.0018226
2: -0.0143622, 0.0043414, -0.0136613, 0.0045676, -0.0143489, 0.0134478
3: -0.0002733, 0.0022018, -0.0001806, 0.0022317, -0.0018988, 0.0017796
4: 0.0028474, 0.0168253, 0.0026783, 0.0163015, -0.0100500, 0.0107235
5: 0.9962974, 1.0001808, 0.9962505, 1.0000353, -0.0027922, 0.0029793
6: 0.0045227, 0.0080478, 0.0044801, 0.0079157, -0.0025345, 0.0027043
7: -0.0065034, 0.0066513, -0.0066625, 0.0061584, -0.0094582, 0.0100920
8: -0.0143696, -0.0041313, -0.0139859, -0.0040074, -0.0078546, 0.0073613
9: -0.0036533, -0.0027700, -0.0036640, -0.0028031, -0.0006351, 0.0006777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A2_B2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018903, upper bound: 0.0020279
time: 1.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019034, upper bound: 0.0020264
time: 1.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0149589, -0.0060544, -0.0146167, -0.0056815, -0.0071515, 0.0064933
1: -0.0071561, -0.0046456, -0.0070597, -0.0045405, -0.0020163, 0.0018307
2: -0.0142398, 0.0042835, -0.0135279, 0.0050591, -0.0148766, 0.0135074
3: -0.0002571, 0.0021941, -0.0001629, 0.0022968, -0.0019687, 0.0017875
4: 0.0028907, 0.0167338, 0.0023110, 0.0162018, -0.0100946, 0.0111178
5: 0.9963094, 1.0001553, 0.9961483, 1.0000075, -0.0028046, 0.0030889
6: 0.0045337, 0.0080247, 0.0043875, 0.0078905, -0.0025457, 0.0028038
7: -0.0064627, 0.0065652, -0.0070082, 0.0060646, -0.0095001, 0.0104631
8: -0.0143026, -0.0041629, -0.0139129, -0.0037384, -0.0081435, 0.0073940
9: -0.0036506, -0.0027758, -0.0036872, -0.0028094, -0.0006379, 0.0007026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A2_B2_A1_B1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018812, upper bound: 0.0020153
time: 2.07 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018959, upper bound: 0.0020183
time: 1.43 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0146279, -0.0059244, -0.0152442, -0.0059578, -0.0064758, 0.0071487
1: -0.0070628, -0.0046090, -0.0072366, -0.0046184, -0.0018258, 0.0020155
2: -0.0135512, 0.0045538, -0.0148332, 0.0044844, -0.0134709, 0.0148708
3: -0.0001660, 0.0022299, -0.0003356, 0.0022207, -0.0017827, 0.0019679
4: 0.0026886, 0.0162192, 0.0027405, 0.0171773, -0.0111135, 0.0100673
5: 0.9962532, 1.0000124, 0.9962677, 1.0002786, -0.0030877, 0.0027970
6: 0.0044827, 0.0078949, 0.0044958, 0.0081365, -0.0028027, 0.0025388
7: -0.0066529, 0.0060810, -0.0066040, 0.0069826, -0.0104591, 0.0094745
8: -0.0139257, -0.0040149, -0.0146274, -0.0040529, -0.0073740, 0.0081403
9: -0.0036633, -0.0028083, -0.0036601, -0.0027478, -0.0007023, 0.0006362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of IS_A2_B2_A1_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019363, upper bound: 0.0019463
time: 1.96 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0019344
time: 2.07 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0150919, -0.0060165, -0.0152442, -0.0059578, -0.0068607, 0.0069666
1: -0.0071936, -0.0046349, -0.0072366, -0.0046184, -0.0019343, 0.0019641
2: -0.0145165, 0.0043623, -0.0148332, 0.0044844, -0.0142716, 0.0144919
3: -0.0002937, 0.0022046, -0.0003356, 0.0022207, -0.0018886, 0.0019178
4: 0.0028318, 0.0169406, 0.0027405, 0.0171773, -0.0108303, 0.0106657
5: 0.9962931, 1.0002128, 0.9962677, 1.0002786, -0.0030090, 0.0029632
6: 0.0045188, 0.0080768, 0.0044958, 0.0081365, -0.0027312, 0.0026897
7: -0.0065182, 0.0067598, -0.0066040, 0.0069826, -0.0101925, 0.0100376
8: -0.0144540, -0.0041198, -0.0146274, -0.0040529, -0.0078123, 0.0079329
9: -0.0036543, -0.0027627, -0.0036601, -0.0027478, -0.0006844, 0.0006740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 233

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019310, upper bound: 0.0021037
time: 1.96 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0021038
time: 2.01 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0150822, -0.0059019, -0.0146939, -0.0059145, -0.0069518, 0.0066450
1: -0.0071909, -0.0046026, -0.0070814, -0.0046062, -0.0019600, 0.0018735
2: -0.0144962, 0.0046006, -0.0136885, 0.0045744, -0.0144613, 0.0138230
3: -0.0002910, 0.0022361, -0.0001842, 0.0022326, -0.0019137, 0.0018293
4: 0.0026537, 0.0169254, 0.0026732, 0.0163218, -0.0103305, 0.0108074
5: 0.9962435, 1.0002086, 0.9962490, 1.0000409, -0.0028701, 0.0030026
6: 0.0044739, 0.0080730, 0.0044788, 0.0079208, -0.0026052, 0.0027255
7: -0.0066858, 0.0067456, -0.0066673, 0.0061775, -0.0097221, 0.0101710
8: -0.0144430, -0.0039893, -0.0140008, -0.0040037, -0.0079161, 0.0075667
9: -0.0036656, -0.0027637, -0.0036643, -0.0028018, -0.0006528, 0.0006830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A2_B2_A2_B1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018903, upper bound: 0.0020525
time: 1.83 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019034, upper bound: 0.0020499
time: 1.74 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0150223, -0.0059306, -0.0146303, -0.0056789, -0.0071177, 0.0066720
1: -0.0071740, -0.0046107, -0.0070635, -0.0045398, -0.0020068, 0.0018811
2: -0.0143715, 0.0045409, -0.0135562, 0.0050646, -0.0148063, 0.0138790
3: -0.0002745, 0.0022282, -0.0001667, 0.0022975, -0.0019594, 0.0018367
4: 0.0026983, 0.0168322, 0.0023069, 0.0162230, -0.0103723, 0.0110653
5: 0.9962559, 1.0001827, 0.9961472, 1.0000135, -0.0028817, 0.0030743
6: 0.0044851, 0.0080495, 0.0043864, 0.0078959, -0.0026157, 0.0027905
7: -0.0066438, 0.0066579, -0.0070121, 0.0060845, -0.0097615, 0.0104137
8: -0.0143747, -0.0040220, -0.0139284, -0.0037353, -0.0081050, 0.0075974
9: -0.0036627, -0.0027696, -0.0036875, -0.0028081, -0.0006555, 0.0006993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018812, upper bound: 0.0020376
time: 1.90 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018959, upper bound: 0.0020417
time: 1.43 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0146912, -0.0058057, -0.0152592, -0.0059540, -0.0065305, 0.0072989
1: -0.0070806, -0.0045755, -0.0072408, -0.0046173, -0.0018412, 0.0020578
2: -0.0136827, 0.0048008, -0.0148645, 0.0044924, -0.0135848, 0.0151832
3: -0.0001834, 0.0022626, -0.0003398, 0.0022218, -0.0017977, 0.0020093
4: 0.0025041, 0.0163175, 0.0027346, 0.0172006, -0.0113470, 0.0101524
5: 0.9962019, 1.0000397, 0.9962659, 1.0002850, -0.0031525, 0.0028206
6: 0.0044362, 0.0079197, 0.0044943, 0.0081424, -0.0028615, 0.0025603
7: -0.0068265, 0.0061734, -0.0066096, 0.0070046, -0.0106787, 0.0095546
8: -0.0139977, -0.0038798, -0.0146445, -0.0040486, -0.0074363, 0.0083113
9: -0.0036750, -0.0028021, -0.0036604, -0.0027463, -0.0007171, 0.0006416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of IS_A2_B2_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019364, upper bound: 0.0019584
time: 1.96 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0019497
time: 2.01 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0151599, -0.0058933, -0.0152592, -0.0059540, -0.0069117, 0.0071122
1: -0.0072128, -0.0046002, -0.0072408, -0.0046173, -0.0019487, 0.0020052
2: -0.0146578, 0.0046185, -0.0148645, 0.0044924, -0.0143778, 0.0147949
3: -0.0003124, 0.0022385, -0.0003398, 0.0022218, -0.0019027, 0.0019579
4: 0.0026403, 0.0170462, 0.0027346, 0.0172006, -0.0110568, 0.0107451
5: 0.9962398, 1.0002421, 0.9962659, 1.0002850, -0.0030719, 0.0029853
6: 0.0044705, 0.0081035, 0.0044943, 0.0081424, -0.0027884, 0.0027098
7: -0.0066983, 0.0068592, -0.0066096, 0.0070046, -0.0104057, 0.0101123
8: -0.0145314, -0.0039795, -0.0146445, -0.0040486, -0.0078704, 0.0080987
9: -0.0036664, -0.0027560, -0.0036604, -0.0027463, -0.0006987, 0.0006790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of IS_A2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019364, upper bound: 0.0021179
time: 1.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0021273
time: 1.94 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.35 seconds
IS_A1_B1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0018949, upper bound: 0.0020076
IS_A1_B1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019063, upper bound: 0.0020033
IS_A1_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0018860, upper bound: 0.0019945
IS_A1_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0018988, upper bound: 0.0019940
IS_A1_B1_A1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019388, upper bound: 0.0019183
IS_A1_B1_A1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019261, upper bound: 0.0019104
IS_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019993, upper bound: 0.0020839
IS_A1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019993, upper bound: 0.0021025
IS_A1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0018949, upper bound: 0.0020366
IS_A1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019063, upper bound: 0.0020333
IS_A1_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0018860, upper bound: 0.0020230
IS_A1_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0018988, upper bound: 0.0020245
IS_A1_B1_A2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019388, upper bound: 0.0019339
IS_A1_B1_A2_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019261, upper bound: 0.0019259
IS_A1_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019993, upper bound: 0.0021042
IS_A1_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019992, upper bound: 0.0021324
IS_A1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0020279, upper bound: 0.0018904
IS_A1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0020263, upper bound: 0.0019033
IS_A1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0020153, upper bound: 0.0018816
IS_A1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0018958
IS_A1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019465, upper bound: 0.0020608
IS_A1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019345, upper bound: 0.0020512
IS_A1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019466, upper bound: 0.0020889
IS_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019345, upper bound: 0.0021040
IS_A1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0020525, upper bound: 0.0018904
IS_A1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0020499, upper bound: 0.0019034
IS_A1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0020376, upper bound: 0.0018816
IS_A1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0020418, upper bound: 0.0018959
IS_A1_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019585, upper bound: 0.0020609
IS_A1_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019496, upper bound: 0.0020512
IS_A1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019584, upper bound: 0.0021041
IS_A1_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019496, upper bound: 0.0021041
IS_A2_B1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0018905, upper bound: 0.0020278
IS_A2_B1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019034, upper bound: 0.0020264
IS_A2_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0018816, upper bound: 0.0020153
IS_A2_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0018959, upper bound: 0.0020182
IS_A2_B1_A1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019364, upper bound: 0.0019465
IS_A2_B1_A1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019230, upper bound: 0.0019344
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019310, upper bound: 0.0021037
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019230, upper bound: 0.0021038
IS_A2_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0018904, upper bound: 0.0020524
IS_A2_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019034, upper bound: 0.0020499
IS_A2_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0018816, upper bound: 0.0020377
IS_A2_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0018959, upper bound: 0.0020418
IS_A2_B1_A2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019364, upper bound: 0.0019584
IS_A2_B1_A2_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0019497
IS_A2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019364, upper bound: 0.0021179
IS_A2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0021274
IS_A2_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0018903, upper bound: 0.0020279
IS_A2_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019034, upper bound: 0.0020264
IS_A2_B2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0018812, upper bound: 0.0020153
IS_A2_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0018959, upper bound: 0.0020183
IS_A2_B2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019363, upper bound: 0.0019463
IS_A2_B2_A1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0019344
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019310, upper bound: 0.0021037
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0021038
IS_A2_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0018903, upper bound: 0.0020525
IS_A2_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019034, upper bound: 0.0020499
IS_A2_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0018812, upper bound: 0.0020376
IS_A2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0018959, upper bound: 0.0020417
IS_A2_B2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019364, upper bound: 0.0019584
IS_A2_B2_A2_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0019497
IS_A2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019364, upper bound: 0.0021179
IS_A2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.35
Output dim: 5, lower bound: -0.0019229, upper bound: 0.0021273

## BFS IS instance: IS_A1_B1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0149127, -0.0063388, -0.0146643, -0.0063604, -0.0065812, 0.0062542
1: -0.0071431, -0.0047258, -0.0070731, -0.0047319, -0.0018555, 0.0017633
2: -0.0141436, 0.0036920, -0.0136269, 0.0036469, -0.0136901, 0.0130099
3: -0.0002444, 0.0021159, -0.0001760, 0.0021099, -0.0018117, 0.0017217
4: 0.0033327, 0.0166619, 0.0033664, 0.0162758, -0.0097228, 0.0102312
5: 0.9964322, 1.0001354, 0.9964415, 1.0000281, -0.0027013, 0.0028425
6: 0.0046451, 0.0080066, 0.0046536, 0.0079092, -0.0024519, 0.0025801
7: -0.0060467, 0.0064976, -0.0060150, 0.0061342, -0.0091502, 0.0096287
8: -0.0142499, -0.0044867, -0.0139671, -0.0045114, -0.0074940, 0.0071216
9: -0.0036226, -0.0027803, -0.0036205, -0.0028047, -0.0006144, 0.0006465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018590, upper bound: 0.0019525
time: 1.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018501, upper bound: 0.0019723
time: 1.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0149200, -0.0063081, -0.0145659, -0.0063109, -0.0066487, 0.0062542
1: -0.0071452, -0.0047172, -0.0070453, -0.0047179, -0.0018745, 0.0017633
2: -0.0141588, 0.0037556, -0.0134222, 0.0037498, -0.0138306, 0.0130101
3: -0.0002464, 0.0021243, -0.0001489, 0.0021235, -0.0018303, 0.0017217
4: 0.0032851, 0.0166733, 0.0032895, 0.0161228, -0.0097229, 0.0103361
5: 0.9964190, 1.0001385, 0.9964201, 0.9999856, -0.0027013, 0.0028717
6: 0.0046331, 0.0080094, 0.0046342, 0.0078706, -0.0024520, 0.0026066
7: -0.0060914, 0.0065083, -0.0060874, 0.0059902, -0.0091504, 0.0097274
8: -0.0142583, -0.0044519, -0.0138550, -0.0044551, -0.0075709, 0.0071217
9: -0.0036256, -0.0027796, -0.0036254, -0.0028144, -0.0006144, 0.0006532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018750, upper bound: 0.0019545
time: 1.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018716, upper bound: 0.0019749
time: 1.48 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0148583, -0.0063669, -0.0146066, -0.0061328, -0.0067595, 0.0062857
1: -0.0071278, -0.0047337, -0.0070568, -0.0046677, -0.0019057, 0.0017722
2: -0.0140304, 0.0036335, -0.0135069, 0.0041203, -0.0140611, 0.0130754
3: -0.0002294, 0.0021081, -0.0001601, 0.0021726, -0.0018608, 0.0017303
4: 0.0033764, 0.0165773, 0.0030126, 0.0161861, -0.0097718, 0.0105084
5: 0.9964443, 1.0001118, 0.9963432, 1.0000032, -0.0027149, 0.0029195
6: 0.0046562, 0.0079852, 0.0045644, 0.0078866, -0.0024643, 0.0026501
7: -0.0060055, 0.0064179, -0.0063479, 0.0060498, -0.0091963, 0.0098895
8: -0.0141880, -0.0045187, -0.0139014, -0.0042523, -0.0076970, 0.0071575
9: -0.0036199, -0.0027857, -0.0036429, -0.0028104, -0.0006175, 0.0006641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018490, upper bound: 0.0019426
time: 1.47 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018379, upper bound: 0.0019580
time: 1.57 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0148656, -0.0063364, -0.0145093, -0.0060667, -0.0068441, 0.0062845
1: -0.0071298, -0.0047251, -0.0070294, -0.0046491, -0.0019296, 0.0017718
2: -0.0140456, 0.0036968, -0.0133045, 0.0042578, -0.0142372, 0.0130730
3: -0.0002314, 0.0021165, -0.0001333, 0.0021907, -0.0018841, 0.0017300
4: 0.0033291, 0.0165887, 0.0029099, 0.0160348, -0.0097700, 0.0106400
5: 0.9964312, 1.0001150, 0.9963148, 0.9999612, -0.0027144, 0.0029561
6: 0.0046442, 0.0079881, 0.0045385, 0.0078484, -0.0024638, 0.0026833
7: -0.0060501, 0.0064287, -0.0064446, 0.0059074, -0.0091946, 0.0100134
8: -0.0141963, -0.0044841, -0.0137906, -0.0041770, -0.0077935, 0.0071562
9: -0.0036229, -0.0027849, -0.0036494, -0.0028199, -0.0006174, 0.0006724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018669, upper bound: 0.0019469
time: 1.47 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018622, upper bound: 0.0019635
time: 1.82 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0148985, -0.0063461, -0.0151396, -0.0062514, -0.0066595, 0.0066853
1: -0.0071391, -0.0047279, -0.0072071, -0.0047011, -0.0018776, 0.0018848
2: -0.0141141, 0.0036767, -0.0146156, 0.0038738, -0.0138530, 0.0139067
3: -0.0002405, 0.0021138, -0.0003068, 0.0021399, -0.0018332, 0.0018403
4: 0.0033442, 0.0166399, 0.0031969, 0.0170146, -0.0103930, 0.0103529
5: 0.9964353, 1.0001293, 0.9963944, 1.0002334, -0.0028875, 0.0028763
6: 0.0046480, 0.0080010, 0.0046109, 0.0080955, -0.0026210, 0.0026109
7: -0.0060359, 0.0064768, -0.0061745, 0.0068295, -0.0097810, 0.0097432
8: -0.0142338, -0.0044951, -0.0145083, -0.0043872, -0.0075832, 0.0076126
9: -0.0036219, -0.0027817, -0.0036312, -0.0027580, -0.0006568, 0.0006542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of IS_A1_B1_A1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0020260
time: 1.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0020336
time: 2.06 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0149345, -0.0062126, -0.0151355, -0.0062570, -0.0067340, 0.0067131
1: -0.0071493, -0.0046902, -0.0072059, -0.0047027, -0.0018986, 0.0018927
2: -0.0141890, 0.0039544, -0.0146071, 0.0038620, -0.0140080, 0.0139646
3: -0.0002504, 0.0021506, -0.0003057, 0.0021384, -0.0018537, 0.0018480
4: 0.0031366, 0.0166959, 0.0032056, 0.0170083, -0.0104363, 0.0104687
5: 0.9963776, 1.0001448, 0.9963969, 1.0002316, -0.0028995, 0.0029085
6: 0.0045957, 0.0080151, 0.0046131, 0.0080939, -0.0026319, 0.0026401
7: -0.0062313, 0.0065295, -0.0061663, 0.0068236, -0.0098217, 0.0098523
8: -0.0142748, -0.0043430, -0.0145037, -0.0043937, -0.0076680, 0.0076442
9: -0.0036350, -0.0027782, -0.0036307, -0.0027584, -0.0006595, 0.0006616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of IS_A1_B1_A1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0020423
time: 1.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0020528
time: 1.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0149960, -0.0062057, -0.0146795, -0.0063574, -0.0065533, 0.0064335
1: -0.0071666, -0.0046883, -0.0070773, -0.0047310, -0.0018476, 0.0018138
2: -0.0143168, 0.0039687, -0.0136584, 0.0036532, -0.0136323, 0.0133830
3: -0.0002673, 0.0021525, -0.0001802, 0.0021107, -0.0018040, 0.0017710
4: 0.0031259, 0.0167914, 0.0033617, 0.0162993, -0.0100016, 0.0101879
5: 0.9963747, 1.0001714, 0.9964402, 1.0000347, -0.0027787, 0.0028305
6: 0.0045930, 0.0080392, 0.0046524, 0.0079151, -0.0025223, 0.0025692
7: -0.0062413, 0.0066194, -0.0060194, 0.0061563, -0.0094126, 0.0095879
8: -0.0143448, -0.0043352, -0.0139843, -0.0045080, -0.0074623, 0.0073259
9: -0.0036357, -0.0027721, -0.0036208, -0.0028032, -0.0006320, 0.0006438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018590, upper bound: 0.0019758
time: 1.89 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018501, upper bound: 0.0020034
time: 1.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0150032, -0.0061712, -0.0145845, -0.0063075, -0.0066243, 0.0064322
1: -0.0071686, -0.0046785, -0.0070506, -0.0047170, -0.0018676, 0.0018135
2: -0.0143319, 0.0040406, -0.0134609, 0.0037570, -0.0137799, 0.0133803
3: -0.0002693, 0.0021620, -0.0001540, 0.0021245, -0.0018236, 0.0017707
4: 0.0030722, 0.0168026, 0.0032842, 0.0161517, -0.0099996, 0.0102982
5: 0.9963598, 1.0001745, 0.9964187, 0.9999936, -0.0027782, 0.0028612
6: 0.0045794, 0.0080420, 0.0046329, 0.0078779, -0.0025218, 0.0025971
7: -0.0062919, 0.0066300, -0.0060924, 0.0060174, -0.0094107, 0.0096918
8: -0.0143530, -0.0042959, -0.0138762, -0.0044512, -0.0075431, 0.0073244
9: -0.0036391, -0.0027714, -0.0036257, -0.0028126, -0.0006319, 0.0006508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018750, upper bound: 0.0019780
time: 2.17 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018716, upper bound: 0.0020067
time: 1.51 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0149401, -0.0062349, -0.0146229, -0.0061294, -0.0068098, 0.0064633
1: -0.0071508, -0.0046965, -0.0070614, -0.0046668, -0.0019199, 0.0018222
2: -0.0142007, 0.0039079, -0.0135408, 0.0041274, -0.0141657, 0.0134449
3: -0.0002519, 0.0021445, -0.0001646, 0.0021735, -0.0018746, 0.0017792
4: 0.0031713, 0.0167046, 0.0030073, 0.0162114, -0.0100479, 0.0105866
5: 0.9963874, 1.0001472, 0.9963417, 1.0000103, -0.0027916, 0.0029413
6: 0.0046044, 0.0080173, 0.0045631, 0.0078929, -0.0025339, 0.0026698
7: -0.0061986, 0.0065377, -0.0063529, 0.0060736, -0.0094562, 0.0099631
8: -0.0142812, -0.0043685, -0.0139199, -0.0042484, -0.0077543, 0.0073598
9: -0.0036328, -0.0027776, -0.0036432, -0.0028088, -0.0006350, 0.0006690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018491, upper bound: 0.0019647
time: 1.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018379, upper bound: 0.0019887
time: 1.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0149473, -0.0062005, -0.0145274, -0.0060637, -0.0068953, 0.0064595
1: -0.0071529, -0.0046868, -0.0070345, -0.0046482, -0.0019440, 0.0018212
2: -0.0142156, 0.0039795, -0.0133422, 0.0042641, -0.0143436, 0.0134371
3: -0.0002539, 0.0021539, -0.0001383, 0.0021916, -0.0018981, 0.0017782
4: 0.0031178, 0.0167157, 0.0029051, 0.0160630, -0.0100420, 0.0107195
5: 0.9963724, 1.0001503, 0.9963133, 0.9999690, -0.0027900, 0.0029782
6: 0.0045909, 0.0080201, 0.0045373, 0.0078555, -0.0025325, 0.0027033
7: -0.0062489, 0.0065482, -0.0064491, 0.0059339, -0.0094507, 0.0100882
8: -0.0142893, -0.0043293, -0.0138112, -0.0041735, -0.0078517, 0.0073555
9: -0.0036362, -0.0027769, -0.0036497, -0.0028182, -0.0006346, 0.0006774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018669, upper bound: 0.0019703
time: 1.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018622, upper bound: 0.0019951
time: 1.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0149809, -0.0062153, -0.0151593, -0.0062483, -0.0066121, 0.0068398
1: -0.0071623, -0.0046910, -0.0072126, -0.0047003, -0.0018642, 0.0019284
2: -0.0142855, 0.0039488, -0.0146566, 0.0038801, -0.0137545, 0.0142282
3: -0.0002632, 0.0021499, -0.0003123, 0.0021408, -0.0018202, 0.0018829
4: 0.0031408, 0.0167680, 0.0031921, 0.0170453, -0.0106333, 0.0102793
5: 0.9963788, 1.0001649, 0.9963931, 1.0002419, -0.0029542, 0.0028559
6: 0.0045967, 0.0080333, 0.0046097, 0.0081032, -0.0026816, 0.0025923
7: -0.0062273, 0.0065974, -0.0061790, 0.0068584, -0.0100071, 0.0096739
8: -0.0143276, -0.0043462, -0.0145307, -0.0043837, -0.0075292, 0.0077886
9: -0.0036348, -0.0027736, -0.0036315, -0.0027561, -0.0006720, 0.0006496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of IS_A1_B1_A2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0020448
time: 1.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0020549
time: 1.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0150181, -0.0060661, -0.0151553, -0.0062533, -0.0066915, 0.0069119
1: -0.0071728, -0.0046489, -0.0072115, -0.0047017, -0.0018866, 0.0019487
2: -0.0143629, 0.0042591, -0.0146482, 0.0038697, -0.0139197, 0.0143781
3: -0.0002734, 0.0021909, -0.0003112, 0.0021394, -0.0018421, 0.0019027
4: 0.0029089, 0.0168258, 0.0031999, 0.0170390, -0.0107453, 0.0104027
5: 0.9963144, 1.0001810, 0.9963953, 1.0002402, -0.0029854, 0.0028902
6: 0.0045382, 0.0080479, 0.0046116, 0.0081017, -0.0027098, 0.0026234
7: -0.0064456, 0.0066518, -0.0061717, 0.0068525, -0.0101125, 0.0097901
8: -0.0143700, -0.0041763, -0.0145262, -0.0043894, -0.0076197, 0.0078706
9: -0.0036494, -0.0027700, -0.0036310, -0.0027565, -0.0006790, 0.0006574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0020685
time: 1.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0020832
time: 1.69 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0146643, -0.0063604, -0.0150095, -0.0060774, -0.0066533, 0.0066955
1: -0.0070731, -0.0047319, -0.0071704, -0.0046521, -0.0018758, 0.0018877
2: -0.0136269, 0.0036469, -0.0143449, 0.0042356, -0.0138403, 0.0139280
3: -0.0001760, 0.0021099, -0.0002710, 0.0021878, -0.0018315, 0.0018431
4: 0.0033664, 0.0162758, 0.0029264, 0.0168123, -0.0104089, 0.0103433
5: 0.9964415, 1.0000281, 0.9963193, 1.0001771, -0.0028919, 0.0028737
6: 0.0046536, 0.0079092, 0.0045427, 0.0080445, -0.0026250, 0.0026084
7: -0.0060150, 0.0061342, -0.0064290, 0.0066391, -0.0097959, 0.0097342
8: -0.0139671, -0.0045114, -0.0143601, -0.0041891, -0.0075762, 0.0076242
9: -0.0036205, -0.0028047, -0.0036483, -0.0027708, -0.0006578, 0.0006536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019736, upper bound: 0.0018509
time: 2.37 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019901, upper bound: 0.0018389
time: 1.66 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0145659, -0.0063109, -0.0150165, -0.0060392, -0.0066593, 0.0068036
1: -0.0070453, -0.0047179, -0.0071724, -0.0046413, -0.0018775, 0.0019182
2: -0.0134222, 0.0037498, -0.0143595, 0.0043151, -0.0138527, 0.0141528
3: -0.0001489, 0.0021235, -0.0002730, 0.0021983, -0.0018332, 0.0018729
4: 0.0032895, 0.0161228, 0.0028670, 0.0168233, -0.0105769, 0.0103526
5: 0.9964201, 0.9999856, 0.9963027, 1.0001802, -0.0029386, 0.0028763
6: 0.0046342, 0.0078706, 0.0045277, 0.0080473, -0.0026674, 0.0026108
7: -0.0060874, 0.0059902, -0.0064849, 0.0066494, -0.0099541, 0.0097430
8: -0.0138550, -0.0044551, -0.0143681, -0.0041456, -0.0075830, 0.0077473
9: -0.0036254, -0.0028144, -0.0036521, -0.0027701, -0.0006684, 0.0006542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020114, upper bound: 0.0018611
time: 1.83 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020115, upper bound: 0.0018896
time: 2.17 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0146066, -0.0061328, -0.0149507, -0.0061051, -0.0066864, 0.0069381
1: -0.0070568, -0.0046677, -0.0071538, -0.0046599, -0.0018851, 0.0019561
2: -0.0135069, 0.0041203, -0.0142226, 0.0041781, -0.0139090, 0.0144327
3: -0.0001601, 0.0021726, -0.0002548, 0.0021802, -0.0018406, 0.0019099
4: 0.0030126, 0.0161861, 0.0029694, 0.0167209, -0.0107861, 0.0103947
5: 0.9963432, 1.0000032, 0.9963312, 1.0001519, -0.0029967, 0.0028880
6: 0.0045644, 0.0078866, 0.0045535, 0.0080214, -0.0027201, 0.0026214
7: -0.0063479, 0.0060498, -0.0063886, 0.0065531, -0.0101509, 0.0097826
8: -0.0139014, -0.0042523, -0.0142932, -0.0042206, -0.0076138, 0.0079005
9: -0.0036429, -0.0028104, -0.0036456, -0.0027766, -0.0006816, 0.0006569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019629, upper bound: 0.0018413
time: 1.65 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019762, upper bound: 0.0018265
time: 2.00 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0145093, -0.0060667, -0.0149577, -0.0060670, -0.0066911, 0.0070557
1: -0.0070294, -0.0046491, -0.0071558, -0.0046492, -0.0018865, 0.0019893
2: -0.0133045, 0.0042578, -0.0142371, 0.0042572, -0.0139189, 0.0146773
3: -0.0001333, 0.0021907, -0.0002568, 0.0021907, -0.0018419, 0.0019423
4: 0.0029099, 0.0160348, 0.0029103, 0.0167318, -0.0109689, 0.0104021
5: 0.9963148, 0.9999612, 0.9963149, 1.0001549, -0.0030475, 0.0028900
6: 0.0045385, 0.0078484, 0.0045386, 0.0080242, -0.0027662, 0.0026233
7: -0.0064446, 0.0059074, -0.0064442, 0.0065634, -0.0103229, 0.0097895
8: -0.0137906, -0.0041770, -0.0143011, -0.0041773, -0.0076192, 0.0080344
9: -0.0036494, -0.0028199, -0.0036493, -0.0027759, -0.0006932, 0.0006573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020035, upper bound: 0.0018444
time: 1.56 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020035, upper bound: 0.0018826
time: 1.87 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0151616, -0.0062268, -0.0145301, -0.0059719, -0.0073599, 0.0064913
1: -0.0072133, -0.0046942, -0.0070352, -0.0046224, -0.0020750, 0.0018301
2: -0.0146613, 0.0039247, -0.0133476, 0.0044551, -0.0153100, 0.0135033
3: -0.0003129, 0.0021467, -0.0001391, 0.0022169, -0.0020260, 0.0017869
4: 0.0031588, 0.0170488, 0.0027624, 0.0160671, -0.0100915, 0.0114417
5: 0.9963838, 1.0002429, 0.9962738, 0.9999701, -0.0028037, 0.0031789
6: 0.0046013, 0.0081041, 0.0045013, 0.0078565, -0.0025449, 0.0028854
7: -0.0062104, 0.0068617, -0.0065834, 0.0059377, -0.0094972, 0.0107679
8: -0.0145333, -0.0043593, -0.0138142, -0.0040690, -0.0083807, 0.0073917
9: -0.0036336, -0.0027559, -0.0036587, -0.0028179, -0.0006377, 0.0007230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018966, upper bound: 0.0020321
time: 2.15 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019173, upper bound: 0.0020304
time: 1.92 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0151054, -0.0062549, -0.0144668, -0.0057349, -0.0075235, 0.0065018
1: -0.0071974, -0.0047021, -0.0070174, -0.0045555, -0.0021212, 0.0018331
2: -0.0145445, 0.0038664, -0.0132161, 0.0049481, -0.0156505, 0.0135250
3: -0.0002974, 0.0021390, -0.0001216, 0.0022821, -0.0020711, 0.0017898
4: 0.0032024, 0.0169615, 0.0023940, 0.0159688, -0.0101078, 0.0116962
5: 0.9963959, 1.0002187, 0.9961713, 0.9999428, -0.0028082, 0.0032496
6: 0.0046123, 0.0080821, 0.0044084, 0.0078318, -0.0025490, 0.0029496
7: -0.0061694, 0.0067795, -0.0069301, 0.0058452, -0.0095125, 0.0110074
8: -0.0144694, -0.0043912, -0.0137422, -0.0037991, -0.0085671, 0.0074036
9: -0.0036309, -0.0027614, -0.0036820, -0.0028241, -0.0006387, 0.0007391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_B1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018861, upper bound: 0.0020175
time: 1.49 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019057, upper bound: 0.0020215
time: 1.96 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0150511, -0.0062733, -0.0150919, -0.0060165, -0.0071022, 0.0068342
1: -0.0071821, -0.0047073, -0.0071936, -0.0046349, -0.0020024, 0.0019268
2: -0.0144315, 0.0038281, -0.0145165, 0.0043623, -0.0147739, 0.0142166
3: -0.0002825, 0.0021339, -0.0002937, 0.0022046, -0.0019551, 0.0018813
4: 0.0032310, 0.0168770, 0.0028318, 0.0169406, -0.0106246, 0.0110411
5: 0.9964039, 1.0001951, 0.9962931, 1.0002128, -0.0029518, 0.0030675
6: 0.0046195, 0.0080608, 0.0045188, 0.0080768, -0.0026794, 0.0027844
7: -0.0061424, 0.0067000, -0.0065182, 0.0067598, -0.0099990, 0.0103909
8: -0.0144075, -0.0044122, -0.0144540, -0.0041198, -0.0080873, 0.0077822
9: -0.0036291, -0.0027667, -0.0036543, -0.0027627, -0.0006714, 0.0006977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020024, upper bound: 0.0020657
time: 1.71 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020083, upper bound: 0.0020657
time: 2.26 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0149988, -0.0060152, -0.0150332, -0.0060441, -0.0072098, 0.0071023
1: -0.0071674, -0.0046346, -0.0071771, -0.0046427, -0.0020327, 0.0020024
2: -0.0143228, 0.0043651, -0.0143943, 0.0043048, -0.0149978, 0.0147742
3: -0.0002681, 0.0022049, -0.0002776, 0.0021970, -0.0019847, 0.0019551
4: 0.0028297, 0.0167958, 0.0028747, 0.0168493, -0.0110413, 0.0112084
5: 0.9962924, 1.0001725, 0.9963049, 1.0001874, -0.0030676, 0.0031140
6: 0.0045183, 0.0080403, 0.0045296, 0.0080538, -0.0027845, 0.0028266
7: -0.0065201, 0.0066236, -0.0064777, 0.0066739, -0.0103911, 0.0105484
8: -0.0143480, -0.0041183, -0.0143872, -0.0041512, -0.0082098, 0.0080874
9: -0.0036544, -0.0027719, -0.0036516, -0.0027685, -0.0006977, 0.0007083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020052, upper bound: 0.0020583
time: 1.75 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020055, upper bound: 0.0020732
time: 1.49 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0146795, -0.0063574, -0.0150739, -0.0059530, -0.0068148, 0.0067524
1: -0.0070773, -0.0047310, -0.0071886, -0.0046170, -0.0019214, 0.0019038
2: -0.0136584, 0.0036532, -0.0144789, 0.0044944, -0.0141762, 0.0140464
3: -0.0001802, 0.0021107, -0.0002888, 0.0022221, -0.0018760, 0.0018588
4: 0.0033617, 0.0162993, 0.0027331, 0.0169125, -0.0104974, 0.0105944
5: 0.9964402, 1.0000347, 0.9962656, 1.0002050, -0.0029165, 0.0029434
6: 0.0046524, 0.0079151, 0.0044939, 0.0080698, -0.0026473, 0.0026718
7: -0.0060194, 0.0061563, -0.0066110, 0.0067334, -0.0098792, 0.0099705
8: -0.0139843, -0.0045080, -0.0144335, -0.0040475, -0.0077601, 0.0076890
9: -0.0036208, -0.0028032, -0.0036605, -0.0027645, -0.0006634, 0.0006695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B2_A1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019922, upper bound: 0.0018511
time: 2.09 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020162, upper bound: 0.0018387
time: 1.61 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0145845, -0.0063075, -0.0150810, -0.0059147, -0.0068173, 0.0068584
1: -0.0070506, -0.0047170, -0.0071905, -0.0046062, -0.0019221, 0.0019336
2: -0.0134609, 0.0037570, -0.0144936, 0.0045741, -0.0141814, 0.0142668
3: -0.0001540, 0.0021245, -0.0002907, 0.0022326, -0.0018767, 0.0018880
4: 0.0032842, 0.0161517, 0.0026735, 0.0169235, -0.0106621, 0.0105983
5: 0.9964187, 0.9999936, 0.9962491, 1.0002080, -0.0029623, 0.0029445
6: 0.0046329, 0.0078779, 0.0044789, 0.0080725, -0.0026888, 0.0026727
7: -0.0060924, 0.0060174, -0.0066671, 0.0067437, -0.0100342, 0.0099742
8: -0.0138762, -0.0044512, -0.0144415, -0.0040038, -0.0077629, 0.0078097
9: -0.0036257, -0.0028126, -0.0036643, -0.0027638, -0.0006738, 0.0006697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020348, upper bound: 0.0018611
time: 1.68 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020348, upper bound: 0.0018896
time: 1.90 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0146229, -0.0061294, -0.0150140, -0.0059816, -0.0068464, 0.0069134
1: -0.0070614, -0.0046668, -0.0071717, -0.0046251, -0.0019303, 0.0019491
2: -0.0135408, 0.0041274, -0.0143543, 0.0044350, -0.0142419, 0.0143813
3: -0.0001646, 0.0021735, -0.0002723, 0.0022142, -0.0018847, 0.0019031
4: 0.0030073, 0.0162114, 0.0027774, 0.0168194, -0.0107477, 0.0106435
5: 0.9963417, 1.0000103, 0.9962778, 1.0001792, -0.0029860, 0.0029571
6: 0.0045631, 0.0078929, 0.0045051, 0.0080463, -0.0027104, 0.0026841
7: -0.0063529, 0.0060736, -0.0065693, 0.0066458, -0.0101148, 0.0100168
8: -0.0139199, -0.0042484, -0.0143653, -0.0040800, -0.0077961, 0.0078724
9: -0.0036432, -0.0028088, -0.0036577, -0.0027704, -0.0006792, 0.0006726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B2_A1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019804, upper bound: 0.0018412
time: 1.65 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019985, upper bound: 0.0018265
time: 1.59 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0145274, -0.0060637, -0.0150210, -0.0059434, -0.0068464, 0.0070361
1: -0.0070345, -0.0046482, -0.0071736, -0.0046143, -0.0019302, 0.0019837
2: -0.0133422, 0.0042641, -0.0143689, 0.0045144, -0.0142418, 0.0146364
3: -0.0001383, 0.0021916, -0.0002742, 0.0022247, -0.0018847, 0.0019369
4: 0.0029051, 0.0160630, 0.0027181, 0.0168303, -0.0109384, 0.0106434
5: 0.9963133, 0.9999690, 0.9962614, 1.0001822, -0.0030390, 0.0029571
6: 0.0045373, 0.0078555, 0.0044901, 0.0080490, -0.0027585, 0.0026841
7: -0.0064491, 0.0059339, -0.0066251, 0.0066560, -0.0102942, 0.0100167
8: -0.0138112, -0.0041735, -0.0143733, -0.0040365, -0.0077960, 0.0080120
9: -0.0036497, -0.0028182, -0.0036615, -0.0027697, -0.0006912, 0.0006726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B2_A1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020271, upper bound: 0.0018446
time: 1.94 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020271, upper bound: 0.0018827
time: 1.83 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0151813, -0.0062230, -0.0145916, -0.0058527, -0.0075077, 0.0064836
1: -0.0072188, -0.0046932, -0.0070526, -0.0045887, -0.0021167, 0.0018280
2: -0.0147023, 0.0039327, -0.0134756, 0.0047031, -0.0156175, 0.0134871
3: -0.0003183, 0.0021477, -0.0001560, 0.0022497, -0.0020667, 0.0017848
4: 0.0031528, 0.0170795, 0.0025771, 0.0161627, -0.0100794, 0.0116715
5: 0.9963822, 1.0002514, 0.9962222, 0.9999967, -0.0028004, 0.0032427
6: 0.0045998, 0.0081119, 0.0044546, 0.0078807, -0.0025419, 0.0029434
7: -0.0062160, 0.0068906, -0.0067578, 0.0060277, -0.0094859, 0.0109842
8: -0.0145558, -0.0043549, -0.0138843, -0.0039332, -0.0085490, 0.0073829
9: -0.0036340, -0.0027539, -0.0036704, -0.0028119, -0.0006370, 0.0007376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_B2_A2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019115, upper bound: 0.0020322
time: 2.06 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019302, upper bound: 0.0020304
time: 1.50 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0151253, -0.0062510, -0.0145274, -0.0056200, -0.0076799, 0.0065481
1: -0.0072030, -0.0047011, -0.0070345, -0.0045231, -0.0021652, 0.0018462
2: -0.0145859, 0.0038744, -0.0133420, 0.0051872, -0.0159757, 0.0136214
3: -0.0003029, 0.0021400, -0.0001383, 0.0023137, -0.0021141, 0.0018026
4: 0.0031964, 0.0169924, 0.0022153, 0.0160628, -0.0101798, 0.0119393
5: 0.9963943, 1.0002273, 0.9961216, 0.9999689, -0.0028283, 0.0033171
6: 0.0046107, 0.0080899, 0.0043633, 0.0078555, -0.0025672, 0.0030109
7: -0.0061750, 0.0068086, -0.0070983, 0.0059338, -0.0095803, 0.0112362
8: -0.0144920, -0.0043868, -0.0138111, -0.0036683, -0.0087451, 0.0074564
9: -0.0036313, -0.0027594, -0.0036933, -0.0028182, -0.0006433, 0.0007545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019009, upper bound: 0.0020175
time: 1.91 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019213, upper bound: 0.0020213
time: 1.74 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0151813, -0.0062230, -0.0150545, -0.0059379, -0.0073123, 0.0069174
1: -0.0072188, -0.0046932, -0.0071831, -0.0046128, -0.0020616, 0.0019503
2: -0.0147023, 0.0039327, -0.0144386, 0.0045258, -0.0152112, 0.0143895
3: -0.0003183, 0.0021477, -0.0002834, 0.0022262, -0.0020130, 0.0019042
4: 0.0031528, 0.0170795, 0.0027096, 0.0168824, -0.0107538, 0.0113679
5: 0.9963822, 1.0002514, 0.9962590, 1.0001967, -0.0029877, 0.0031583
6: 0.0045998, 0.0081119, 0.0044880, 0.0080622, -0.0027120, 0.0028668
7: -0.0062160, 0.0068906, -0.0066331, 0.0067051, -0.0101205, 0.0106984
8: -0.0145558, -0.0043549, -0.0144114, -0.0040303, -0.0083266, 0.0078768
9: -0.0036340, -0.0027539, -0.0036620, -0.0027664, -0.0006796, 0.0007184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_B2_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020112, upper bound: 0.0020797
time: 1.58 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020220, upper bound: 0.0020797
time: 1.65 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0151253, -0.0062510, -0.0149889, -0.0056956, -0.0074807, 0.0069254
1: -0.0072030, -0.0047011, -0.0071646, -0.0045445, -0.0021091, 0.0019525
2: -0.0145859, 0.0038744, -0.0143021, 0.0050299, -0.0155614, 0.0144062
3: -0.0003029, 0.0021400, -0.0002654, 0.0022929, -0.0020593, 0.0019064
4: 0.0031964, 0.0169924, 0.0023328, 0.0167804, -0.0107663, 0.0116296
5: 0.9963943, 1.0002273, 0.9961544, 1.0001683, -0.0029912, 0.0032310
6: 0.0046107, 0.0080899, 0.0043930, 0.0080364, -0.0027151, 0.0029328
7: -0.0061750, 0.0068086, -0.0069877, 0.0066091, -0.0101323, 0.0109448
8: -0.0144920, -0.0043868, -0.0143367, -0.0037543, -0.0085183, 0.0078860
9: -0.0036313, -0.0027594, -0.0036858, -0.0027728, -0.0006804, 0.0007349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_B2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020044, upper bound: 0.0020730
time: 1.65 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020190, upper bound: 0.0020728
time: 2.11 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0150095, -0.0060774, -0.0146643, -0.0063604, -0.0066955, 0.0066533
1: -0.0071704, -0.0046521, -0.0070731, -0.0047319, -0.0018877, 0.0018758
2: -0.0143449, 0.0042356, -0.0136269, 0.0036469, -0.0139280, 0.0138403
3: -0.0002710, 0.0021878, -0.0001760, 0.0021099, -0.0018431, 0.0018315
4: 0.0029264, 0.0168123, 0.0033664, 0.0162758, -0.0103433, 0.0104089
5: 0.9963193, 1.0001771, 0.9964415, 1.0000281, -0.0028737, 0.0028919
6: 0.0045427, 0.0080445, 0.0046536, 0.0079092, -0.0026084, 0.0026250
7: -0.0064290, 0.0066391, -0.0060150, 0.0061342, -0.0097342, 0.0097959
8: -0.0143601, -0.0041891, -0.0139671, -0.0045114, -0.0076242, 0.0075762
9: -0.0036483, -0.0027708, -0.0036205, -0.0028047, -0.0006536, 0.0006578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A1_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018511, upper bound: 0.0019736
time: 2.12 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018389, upper bound: 0.0019902
time: 1.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0150165, -0.0060392, -0.0145659, -0.0063109, -0.0068036, 0.0066593
1: -0.0071724, -0.0046413, -0.0070453, -0.0047179, -0.0019182, 0.0018775
2: -0.0143595, 0.0043151, -0.0134222, 0.0037498, -0.0141528, 0.0138527
3: -0.0002730, 0.0021983, -0.0001489, 0.0021235, -0.0018729, 0.0018332
4: 0.0028670, 0.0168233, 0.0032895, 0.0161228, -0.0103526, 0.0105769
5: 0.9963027, 1.0001802, 0.9964201, 0.9999856, -0.0028763, 0.0029386
6: 0.0045277, 0.0080473, 0.0046342, 0.0078706, -0.0026108, 0.0026674
7: -0.0064849, 0.0066494, -0.0060874, 0.0059902, -0.0097430, 0.0099541
8: -0.0143681, -0.0041456, -0.0138550, -0.0044551, -0.0077473, 0.0075830
9: -0.0036521, -0.0027701, -0.0036254, -0.0028144, -0.0006542, 0.0006684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018612, upper bound: 0.0020114
time: 2.15 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018896, upper bound: 0.0020115
time: 1.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0149507, -0.0061051, -0.0146066, -0.0061328, -0.0069381, 0.0066864
1: -0.0071538, -0.0046599, -0.0070568, -0.0046677, -0.0019561, 0.0018851
2: -0.0142226, 0.0041781, -0.0135069, 0.0041203, -0.0144327, 0.0139090
3: -0.0002548, 0.0021802, -0.0001601, 0.0021726, -0.0019099, 0.0018406
4: 0.0029694, 0.0167209, 0.0030126, 0.0161861, -0.0103947, 0.0107861
5: 0.9963312, 1.0001519, 0.9963432, 1.0000032, -0.0028880, 0.0029967
6: 0.0045535, 0.0080214, 0.0045644, 0.0078866, -0.0026214, 0.0027201
7: -0.0063886, 0.0065531, -0.0063479, 0.0060498, -0.0097826, 0.0101509
8: -0.0142932, -0.0042206, -0.0139014, -0.0042523, -0.0079005, 0.0076138
9: -0.0036456, -0.0027766, -0.0036429, -0.0028104, -0.0006569, 0.0006816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018413, upper bound: 0.0019629
time: 1.68 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018263, upper bound: 0.0019762
time: 1.49 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0149577, -0.0060670, -0.0145093, -0.0060667, -0.0070557, 0.0066911
1: -0.0071558, -0.0046492, -0.0070294, -0.0046491, -0.0019893, 0.0018865
2: -0.0142371, 0.0042572, -0.0133045, 0.0042578, -0.0146773, 0.0139189
3: -0.0002568, 0.0021907, -0.0001333, 0.0021907, -0.0019423, 0.0018419
4: 0.0029103, 0.0167318, 0.0029099, 0.0160348, -0.0104021, 0.0109689
5: 0.9963149, 1.0001549, 0.9963148, 0.9999612, -0.0028900, 0.0030475
6: 0.0045386, 0.0080242, 0.0045385, 0.0078484, -0.0026233, 0.0027662
7: -0.0064442, 0.0065634, -0.0064446, 0.0059074, -0.0097895, 0.0103229
8: -0.0143011, -0.0041773, -0.0137906, -0.0041770, -0.0080344, 0.0076192
9: -0.0036493, -0.0027759, -0.0036494, -0.0028199, -0.0006573, 0.0006932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B1_B2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018445, upper bound: 0.0020035
time: 1.93 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018828, upper bound: 0.0020035
time: 1.98 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0150919, -0.0060165, -0.0150511, -0.0062733, -0.0068342, 0.0071022
1: -0.0071936, -0.0046349, -0.0071821, -0.0047073, -0.0019268, 0.0020024
2: -0.0145165, 0.0043623, -0.0144315, 0.0038281, -0.0142166, 0.0147739
3: -0.0002937, 0.0022046, -0.0002825, 0.0021339, -0.0018813, 0.0019551
4: 0.0028318, 0.0169406, 0.0032310, 0.0168770, -0.0110411, 0.0106246
5: 0.9962931, 1.0002128, 0.9964039, 1.0001951, -0.0030675, 0.0029518
6: 0.0045188, 0.0080768, 0.0046195, 0.0080608, -0.0027844, 0.0026794
7: -0.0065182, 0.0067598, -0.0061424, 0.0067000, -0.0103909, 0.0099990
8: -0.0144540, -0.0041198, -0.0144075, -0.0044122, -0.0077822, 0.0080873
9: -0.0036543, -0.0027627, -0.0036291, -0.0027667, -0.0006977, 0.0006714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019939, upper bound: 0.0020599
time: 1.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019939, upper bound: 0.0020768
time: 2.20 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0150332, -0.0060441, -0.0149988, -0.0060152, -0.0071023, 0.0072098
1: -0.0071771, -0.0046427, -0.0071674, -0.0046346, -0.0020024, 0.0020327
2: -0.0143943, 0.0043048, -0.0143228, 0.0043651, -0.0147742, 0.0149978
3: -0.0002776, 0.0021970, -0.0002681, 0.0022049, -0.0019551, 0.0019847
4: 0.0028747, 0.0168493, 0.0028297, 0.0167958, -0.0112084, 0.0110413
5: 0.9963049, 1.0001874, 0.9962924, 1.0001725, -0.0031140, 0.0030676
6: 0.0045296, 0.0080538, 0.0045183, 0.0080403, -0.0028266, 0.0027845
7: -0.0064777, 0.0066739, -0.0065201, 0.0066236, -0.0105484, 0.0103911
8: -0.0143872, -0.0041512, -0.0143480, -0.0041183, -0.0080874, 0.0082098
9: -0.0036516, -0.0027685, -0.0036544, -0.0027719, -0.0007083, 0.0006977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019815, upper bound: 0.0020719
time: 2.24 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019902, upper bound: 0.0020721
time: 1.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0150739, -0.0059530, -0.0146795, -0.0063574, -0.0067524, 0.0068148
1: -0.0071886, -0.0046170, -0.0070773, -0.0047310, -0.0019038, 0.0019214
2: -0.0144789, 0.0044944, -0.0136584, 0.0036532, -0.0140464, 0.0141762
3: -0.0002888, 0.0022221, -0.0001802, 0.0021107, -0.0018588, 0.0018760
4: 0.0027331, 0.0169125, 0.0033617, 0.0162993, -0.0105944, 0.0104974
5: 0.9962656, 1.0002050, 0.9964402, 1.0000347, -0.0029434, 0.0029165
6: 0.0044939, 0.0080698, 0.0046524, 0.0079151, -0.0026718, 0.0026473
7: -0.0066110, 0.0067334, -0.0060194, 0.0061563, -0.0099705, 0.0098792
8: -0.0144335, -0.0040475, -0.0139843, -0.0045080, -0.0076890, 0.0077601
9: -0.0036605, -0.0027645, -0.0036208, -0.0028032, -0.0006695, 0.0006634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018509, upper bound: 0.0019922
time: 1.99 seconds

## Relational analysis of IS_A2_B1_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018389, upper bound: 0.0020162
time: 1.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0150810, -0.0059147, -0.0145845, -0.0063075, -0.0068584, 0.0068173
1: -0.0071905, -0.0046062, -0.0070506, -0.0047170, -0.0019336, 0.0019221
2: -0.0144936, 0.0045741, -0.0134609, 0.0037570, -0.0142668, 0.0141814
3: -0.0002907, 0.0022326, -0.0001540, 0.0021245, -0.0018880, 0.0018767
4: 0.0026735, 0.0169235, 0.0032842, 0.0161517, -0.0105983, 0.0106621
5: 0.9962491, 1.0002080, 0.9964187, 0.9999936, -0.0029445, 0.0029623
6: 0.0044789, 0.0080725, 0.0046329, 0.0078779, -0.0026727, 0.0026888
7: -0.0066671, 0.0067437, -0.0060924, 0.0060174, -0.0099742, 0.0100342
8: -0.0144415, -0.0040038, -0.0138762, -0.0044512, -0.0078097, 0.0077629
9: -0.0036643, -0.0027638, -0.0036257, -0.0028126, -0.0006697, 0.0006738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018612, upper bound: 0.0020348
time: 1.98 seconds

## Relational analysis of IS_A2_B1_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_B1_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018896, upper bound: 0.0020347
time: 1.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0150140, -0.0059816, -0.0146229, -0.0061294, -0.0069134, 0.0068464
1: -0.0071717, -0.0046251, -0.0070614, -0.0046668, -0.0019491, 0.0019303
2: -0.0143543, 0.0044350, -0.0135408, 0.0041274, -0.0143813, 0.0142419
3: -0.0002723, 0.0022142, -0.0001646, 0.0021735, -0.0019031, 0.0018847
4: 0.0027774, 0.0168194, 0.0030073, 0.0162114, -0.0106435, 0.0107477
5: 0.9962778, 1.0001792, 0.9963417, 1.0000103, -0.0029571, 0.0029860
6: 0.0045051, 0.0080463, 0.0045631, 0.0078929, -0.0026841, 0.0027104
7: -0.0065693, 0.0066458, -0.0063529, 0.0060736, -0.0100168, 0.0101148
8: -0.0143653, -0.0040800, -0.0139199, -0.0042484, -0.0078724, 0.0077961
9: -0.0036577, -0.0027704, -0.0036432, -0.0028088, -0.0006726, 0.0006792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018412, upper bound: 0.0019804
time: 1.92 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018263, upper bound: 0.0019986
time: 1.87 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0150210, -0.0059434, -0.0145274, -0.0060637, -0.0070361, 0.0068464
1: -0.0071736, -0.0046143, -0.0070345, -0.0046482, -0.0019837, 0.0019302
2: -0.0143689, 0.0045144, -0.0133422, 0.0042641, -0.0146364, 0.0142418
3: -0.0002742, 0.0022247, -0.0001383, 0.0021916, -0.0019369, 0.0018847
4: 0.0027181, 0.0168303, 0.0029051, 0.0160630, -0.0106434, 0.0109384
5: 0.9962614, 1.0001822, 0.9963133, 0.9999690, -0.0029571, 0.0030390
6: 0.0044901, 0.0080490, 0.0045373, 0.0078555, -0.0026841, 0.0027585
7: -0.0066251, 0.0066560, -0.0064491, 0.0059339, -0.0100167, 0.0102942
8: -0.0143733, -0.0040365, -0.0138112, -0.0041735, -0.0080120, 0.0077960
9: -0.0036615, -0.0027697, -0.0036497, -0.0028182, -0.0006726, 0.0006912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018446, upper bound: 0.0020271
time: 1.97 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018827, upper bound: 0.0020271
time: 1.82 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0150545, -0.0059379, -0.0151813, -0.0062230, -0.0069174, 0.0073123
1: -0.0071831, -0.0046128, -0.0072188, -0.0046932, -0.0019503, 0.0020616
2: -0.0144386, 0.0045258, -0.0147023, 0.0039327, -0.0143895, 0.0152112
3: -0.0002834, 0.0022262, -0.0003183, 0.0021477, -0.0019042, 0.0020130
4: 0.0027096, 0.0168824, 0.0031528, 0.0170795, -0.0113679, 0.0107538
5: 0.9962590, 1.0001967, 0.9963822, 1.0002514, -0.0031583, 0.0029877
6: 0.0044880, 0.0080622, 0.0045998, 0.0081119, -0.0028668, 0.0027120
7: -0.0066331, 0.0067051, -0.0062160, 0.0068906, -0.0106984, 0.0101205
8: -0.0144114, -0.0040303, -0.0145558, -0.0043549, -0.0078768, 0.0083266
9: -0.0036620, -0.0027664, -0.0036340, -0.0027539, -0.0007184, 0.0006796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019947, upper bound: 0.0020722
time: 1.68 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019948, upper bound: 0.0020934
time: 2.27 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0149889, -0.0056956, -0.0151253, -0.0062510, -0.0069254, 0.0074807
1: -0.0071646, -0.0045445, -0.0072030, -0.0047011, -0.0019525, 0.0021091
2: -0.0143021, 0.0050299, -0.0145859, 0.0038744, -0.0144062, 0.0155614
3: -0.0002654, 0.0022929, -0.0003029, 0.0021400, -0.0019064, 0.0020593
4: 0.0023328, 0.0167804, 0.0031964, 0.0169924, -0.0116296, 0.0107663
5: 0.9961544, 1.0001683, 0.9963943, 1.0002273, -0.0032310, 0.0029912
6: 0.0043930, 0.0080364, 0.0046107, 0.0080899, -0.0029328, 0.0027151
7: -0.0069877, 0.0066091, -0.0061750, 0.0068086, -0.0109448, 0.0101323
8: -0.0143367, -0.0037543, -0.0144920, -0.0043868, -0.0078860, 0.0085183
9: -0.0036858, -0.0027728, -0.0036313, -0.0027594, -0.0007349, 0.0006804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A2_B1_A2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019897, upper bound: 0.0020719
time: 1.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019902, upper bound: 0.0020946
time: 1.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0150095, -0.0060774, -0.0147793, -0.0060861, -0.0066835, 0.0064368
1: -0.0071704, -0.0046521, -0.0071055, -0.0046546, -0.0018843, 0.0018148
2: -0.0143449, 0.0042356, -0.0138662, 0.0042175, -0.0139031, 0.0133899
3: -0.0002710, 0.0021878, -0.0002077, 0.0021854, -0.0018399, 0.0017719
4: 0.0029264, 0.0168123, 0.0029400, 0.0164546, -0.0100068, 0.0103903
5: 0.9963193, 1.0001771, 0.9963230, 1.0000777, -0.0027802, 0.0028867
6: 0.0045427, 0.0080445, 0.0045461, 0.0079543, -0.0025236, 0.0026203
7: -0.0064290, 0.0066391, -0.0064163, 0.0063024, -0.0094175, 0.0097785
8: -0.0143601, -0.0041891, -0.0140981, -0.0041991, -0.0076106, 0.0073297
9: -0.0036483, -0.0027708, -0.0036475, -0.0027934, -0.0006324, 0.0006566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018500, upper bound: 0.0019736
time: 1.90 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018384, upper bound: 0.0019901
time: 2.00 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0150165, -0.0060392, -0.0146700, -0.0060379, -0.0067685, 0.0064323
1: -0.0071724, -0.0046413, -0.0070747, -0.0046410, -0.0019083, 0.0018135
2: -0.0143595, 0.0043151, -0.0136387, 0.0043177, -0.0140799, 0.0133806
3: -0.0002730, 0.0021983, -0.0001776, 0.0021987, -0.0018633, 0.0017707
4: 0.0028670, 0.0168233, 0.0028651, 0.0162846, -0.0099998, 0.0105224
5: 0.9963027, 1.0001802, 0.9963022, 1.0000305, -0.0027782, 0.0029234
6: 0.0045277, 0.0080473, 0.0045272, 0.0079114, -0.0025218, 0.0026536
7: -0.0064849, 0.0066494, -0.0064868, 0.0061424, -0.0094109, 0.0099028
8: -0.0143681, -0.0041456, -0.0139735, -0.0041442, -0.0077074, 0.0073245
9: -0.0036521, -0.0027701, -0.0036522, -0.0028042, -0.0006319, 0.0006650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018610, upper bound: 0.0020115
time: 2.16 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018896, upper bound: 0.0020115
time: 1.57 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0149507, -0.0061051, -0.0147153, -0.0058580, -0.0069421, 0.0064669
1: -0.0071538, -0.0046599, -0.0070874, -0.0045903, -0.0019572, 0.0018232
2: -0.0142226, 0.0041781, -0.0137330, 0.0046919, -0.0144410, 0.0134524
3: -0.0002548, 0.0021802, -0.0001900, 0.0022482, -0.0019110, 0.0017802
4: 0.0029694, 0.0167209, 0.0025854, 0.0163550, -0.0100535, 0.0107923
5: 0.9963312, 1.0001519, 0.9962245, 1.0000501, -0.0027932, 0.0029984
6: 0.0045535, 0.0080214, 0.0044567, 0.0079292, -0.0025353, 0.0027217
7: -0.0063886, 0.0065531, -0.0067500, 0.0062088, -0.0094614, 0.0101568
8: -0.0142932, -0.0042206, -0.0140252, -0.0039393, -0.0079050, 0.0073639
9: -0.0036456, -0.0027766, -0.0036699, -0.0027997, -0.0006353, 0.0006820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018401, upper bound: 0.0019629
time: 1.92 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018256, upper bound: 0.0019762
time: 1.89 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0149577, -0.0060670, -0.0146052, -0.0058025, -0.0070349, 0.0064619
1: -0.0071558, -0.0046492, -0.0070564, -0.0045746, -0.0019834, 0.0018219
2: -0.0142371, 0.0042572, -0.0135040, 0.0048076, -0.0146341, 0.0134421
3: -0.0002568, 0.0021907, -0.0001597, 0.0022635, -0.0019366, 0.0017788
4: 0.0029103, 0.0167318, 0.0024990, 0.0161839, -0.0100458, 0.0109366
5: 0.9963149, 1.0001549, 0.9962005, 1.0000026, -0.0027910, 0.0030385
6: 0.0045386, 0.0080242, 0.0044349, 0.0078860, -0.0025334, 0.0027580
7: -0.0064442, 0.0065634, -0.0068313, 0.0060477, -0.0094542, 0.0102925
8: -0.0143011, -0.0041773, -0.0138998, -0.0038761, -0.0080107, 0.0073582
9: -0.0036493, -0.0027759, -0.0036753, -0.0028105, -0.0006348, 0.0006911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018443, upper bound: 0.0020035
time: 1.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018828, upper bound: 0.0020034
time: 1.81 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0150919, -0.0060165, -0.0151374, -0.0060032, -0.0068155, 0.0068526
1: -0.0071936, -0.0046349, -0.0072064, -0.0046312, -0.0019215, 0.0019320
2: -0.0145165, 0.0043623, -0.0146109, 0.0043899, -0.0141777, 0.0142549
3: -0.0002937, 0.0022046, -0.0003062, 0.0022082, -0.0018762, 0.0018864
4: 0.0028318, 0.0169406, 0.0028111, 0.0170112, -0.0106532, 0.0105955
5: 0.9962931, 1.0002128, 0.9962872, 1.0002325, -0.0029598, 0.0029437
6: 0.0045188, 0.0080768, 0.0045136, 0.0080946, -0.0026866, 0.0026720
7: -0.0065182, 0.0067598, -0.0065375, 0.0068263, -0.0100258, 0.0099715
8: -0.0144540, -0.0041198, -0.0145058, -0.0041047, -0.0077609, 0.0078031
9: -0.0036543, -0.0027627, -0.0036556, -0.0027582, -0.0006732, 0.0006696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019795, upper bound: 0.0020724
time: 2.28 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019891, upper bound: 0.0020719
time: 1.96 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0150332, -0.0060441, -0.0150685, -0.0057562, -0.0070929, 0.0068778
1: -0.0071771, -0.0046427, -0.0071870, -0.0045615, -0.0019997, 0.0019391
2: -0.0143943, 0.0043048, -0.0144676, 0.0049039, -0.0147547, 0.0143073
3: -0.0002776, 0.0021970, -0.0002873, 0.0022762, -0.0019525, 0.0018933
4: 0.0028747, 0.0168493, 0.0024270, 0.0169041, -0.0106924, 0.0110267
5: 0.9963049, 1.0001874, 0.9961805, 1.0002028, -0.0029707, 0.0030635
6: 0.0045296, 0.0080538, 0.0044167, 0.0080676, -0.0026965, 0.0027808
7: -0.0064777, 0.0066739, -0.0068990, 0.0067255, -0.0100627, 0.0103774
8: -0.0143872, -0.0041512, -0.0144273, -0.0038233, -0.0080767, 0.0078318
9: -0.0036516, -0.0027685, -0.0036799, -0.0027650, -0.0006757, 0.0006968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019814, upper bound: 0.0020719
time: 1.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019903, upper bound: 0.0020720
time: 2.16 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0150739, -0.0059530, -0.0147939, -0.0060836, -0.0067369, 0.0066218
1: -0.0071886, -0.0046170, -0.0071096, -0.0046539, -0.0018994, 0.0018669
2: -0.0144789, 0.0044944, -0.0138965, 0.0042226, -0.0140140, 0.0137747
3: -0.0002888, 0.0022221, -0.0002117, 0.0021861, -0.0018545, 0.0018229
4: 0.0027331, 0.0169125, 0.0029361, 0.0164772, -0.0102944, 0.0104732
5: 0.9962656, 1.0002050, 0.9963220, 1.0000842, -0.0028601, 0.0029098
6: 0.0044939, 0.0080698, 0.0045451, 0.0079600, -0.0025961, 0.0026412
7: -0.0066110, 0.0067334, -0.0064199, 0.0063238, -0.0096881, 0.0098565
8: -0.0144335, -0.0040475, -0.0141147, -0.0041962, -0.0076713, 0.0075403
9: -0.0036605, -0.0027645, -0.0036477, -0.0027920, -0.0006505, 0.0006618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018501, upper bound: 0.0019922
time: 2.02 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018384, upper bound: 0.0020162
time: 1.93 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0150810, -0.0059147, -0.0146830, -0.0060347, -0.0068212, 0.0066131
1: -0.0071905, -0.0046062, -0.0070784, -0.0046401, -0.0019232, 0.0018645
2: -0.0144936, 0.0045741, -0.0136659, 0.0043244, -0.0141895, 0.0137565
3: -0.0002907, 0.0022326, -0.0001812, 0.0021996, -0.0018778, 0.0018205
4: 0.0026735, 0.0169235, 0.0028600, 0.0163049, -0.0102808, 0.0106043
5: 0.9962491, 1.0002080, 0.9963008, 1.0000362, -0.0028563, 0.0029462
6: 0.0044789, 0.0080725, 0.0045259, 0.0079165, -0.0025927, 0.0026743
7: -0.0066671, 0.0067437, -0.0064915, 0.0061616, -0.0096753, 0.0099799
8: -0.0144415, -0.0040038, -0.0139884, -0.0041405, -0.0077673, 0.0075303
9: -0.0036643, -0.0027638, -0.0036525, -0.0028029, -0.0006497, 0.0006701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018609, upper bound: 0.0020348
time: 2.04 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018896, upper bound: 0.0020348
time: 1.94 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0150140, -0.0059816, -0.0147301, -0.0058551, -0.0068958, 0.0066506
1: -0.0071717, -0.0046251, -0.0070916, -0.0045894, -0.0019442, 0.0018750
2: -0.0143543, 0.0044350, -0.0137638, 0.0046981, -0.0143446, 0.0138345
3: -0.0002723, 0.0022142, -0.0001941, 0.0022490, -0.0018983, 0.0018308
4: 0.0027774, 0.0168194, 0.0025808, 0.0163781, -0.0103391, 0.0107202
5: 0.9962778, 1.0001792, 0.9962233, 1.0000565, -0.0028725, 0.0029784
6: 0.0045051, 0.0080463, 0.0044555, 0.0079350, -0.0026074, 0.0027035
7: -0.0065693, 0.0066458, -0.0067543, 0.0062304, -0.0097302, 0.0100889
8: -0.0143653, -0.0040800, -0.0140420, -0.0039360, -0.0078522, 0.0075730
9: -0.0036577, -0.0027704, -0.0036702, -0.0027983, -0.0006534, 0.0006775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018401, upper bound: 0.0019804
time: 1.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018256, upper bound: 0.0019986
time: 1.97 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0150210, -0.0059434, -0.0146189, -0.0057998, -0.0069894, 0.0066407
1: -0.0071736, -0.0046143, -0.0070603, -0.0045738, -0.0019706, 0.0018723
2: -0.0143689, 0.0045144, -0.0135323, 0.0048130, -0.0145394, 0.0138140
3: -0.0002742, 0.0022247, -0.0001635, 0.0022642, -0.0019241, 0.0018281
4: 0.0027181, 0.0168303, 0.0024949, 0.0162051, -0.0103237, 0.0108658
5: 0.9962614, 1.0001822, 0.9961994, 1.0000085, -0.0028682, 0.0030189
6: 0.0044901, 0.0080490, 0.0044339, 0.0078914, -0.0026035, 0.0027402
7: -0.0066251, 0.0066560, -0.0068351, 0.0060677, -0.0097158, 0.0102260
8: -0.0143733, -0.0040365, -0.0139153, -0.0038731, -0.0079589, 0.0075618
9: -0.0036615, -0.0027697, -0.0036756, -0.0028092, -0.0006524, 0.0006867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018443, upper bound: 0.0020271
time: 2.13 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018828, upper bound: 0.0020271
time: 1.81 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0150545, -0.0059379, -0.0152592, -0.0059540, -0.0069090, 0.0070671
1: -0.0071831, -0.0046128, -0.0072408, -0.0046173, -0.0019479, 0.0019925
2: -0.0144386, 0.0045258, -0.0148645, 0.0044924, -0.0143720, 0.0147010
3: -0.0002834, 0.0022262, -0.0003398, 0.0022218, -0.0019019, 0.0019454
4: 0.0027096, 0.0168824, 0.0027346, 0.0172006, -0.0109866, 0.0107408
5: 0.9962590, 1.0001967, 0.9962659, 1.0002850, -0.0030524, 0.0029841
6: 0.0044880, 0.0080622, 0.0044943, 0.0081424, -0.0027707, 0.0027087
7: -0.0066331, 0.0067051, -0.0066096, 0.0070046, -0.0103396, 0.0101083
8: -0.0144114, -0.0040303, -0.0146445, -0.0040486, -0.0078673, 0.0080473
9: -0.0036620, -0.0027664, -0.0036604, -0.0027463, -0.0006943, 0.0006788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019948, upper bound: 0.0020722
time: 1.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019948, upper bound: 0.0020934
time: 1.98 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0149889, -0.0056956, -0.0151991, -0.0059815, -0.0069148, 0.0072467
1: -0.0071646, -0.0045445, -0.0072239, -0.0046251, -0.0019495, 0.0020431
2: -0.0143021, 0.0050299, -0.0147394, 0.0044352, -0.0143843, 0.0150746
3: -0.0002654, 0.0022929, -0.0003232, 0.0022142, -0.0019035, 0.0019949
4: 0.0023328, 0.0167804, 0.0027773, 0.0171072, -0.0112659, 0.0107499
5: 0.9961544, 1.0001683, 0.9962778, 1.0002592, -0.0031300, 0.0029866
6: 0.0043930, 0.0080364, 0.0045051, 0.0081188, -0.0028411, 0.0027110
7: -0.0069877, 0.0066091, -0.0065694, 0.0069166, -0.0106024, 0.0101169
8: -0.0143367, -0.0037543, -0.0145761, -0.0040799, -0.0078740, 0.0082519
9: -0.0036858, -0.0027728, -0.0036577, -0.0027522, -0.0007119, 0.0006793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A2_B2_A2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019898, upper bound: 0.0020719
time: 2.29 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019903, upper bound: 0.0020945
time: 2.01 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.87 seconds
IS_A1_B1_A1_B1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018590, upper bound: 0.0019525
IS_A1_B1_A1_B1_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018501, upper bound: 0.0019723
IS_A1_B1_A1_B1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018750, upper bound: 0.0019545
IS_A1_B1_A1_B1_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018716, upper bound: 0.0019749
IS_A1_B1_A1_B1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018490, upper bound: 0.0019426
IS_A1_B1_A1_B1_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018379, upper bound: 0.0019580
IS_A1_B1_A1_B1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018669, upper bound: 0.0019469
IS_A1_B1_A1_B1_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018622, upper bound: 0.0019635
IS_A1_B1_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0020260
IS_A1_B1_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0020336
IS_A1_B1_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0020423
IS_A1_B1_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0020528
IS_A1_B1_A2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018590, upper bound: 0.0019758
IS_A1_B1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018501, upper bound: 0.0020034
IS_A1_B1_A2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018750, upper bound: 0.0019780
IS_A1_B1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018716, upper bound: 0.0020067
IS_A1_B1_A2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018491, upper bound: 0.0019647
IS_A1_B1_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018379, upper bound: 0.0019887
IS_A1_B1_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018669, upper bound: 0.0019703
IS_A1_B1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018622, upper bound: 0.0019951
IS_A1_B1_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0020448
IS_A1_B1_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0020549
IS_A1_B1_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0020685
IS_A1_B1_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0020832
IS_A1_B2_B1_A1_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019736, upper bound: 0.0018509
IS_A1_B2_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019901, upper bound: 0.0018389
IS_A1_B2_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0020114, upper bound: 0.0018611
IS_A1_B2_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0020115, upper bound: 0.0018896
IS_A1_B2_B1_A1_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019629, upper bound: 0.0018413
IS_A1_B2_B1_A1_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019762, upper bound: 0.0018265
IS_A1_B2_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0020035, upper bound: 0.0018444
IS_A1_B2_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0020035, upper bound: 0.0018826
IS_A1_B2_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018966, upper bound: 0.0020321
IS_A1_B2_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019173, upper bound: 0.0020304
IS_A1_B2_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018861, upper bound: 0.0020175
IS_A1_B2_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019057, upper bound: 0.0020215
IS_A1_B2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0020024, upper bound: 0.0020657
IS_A1_B2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0020083, upper bound: 0.0020657
IS_A1_B2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0020052, upper bound: 0.0020583
IS_A1_B2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0020055, upper bound: 0.0020732
IS_A1_B2_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019922, upper bound: 0.0018511
IS_A1_B2_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0020162, upper bound: 0.0018387
IS_A1_B2_B2_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0020348, upper bound: 0.0018611
IS_A1_B2_B2_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0020348, upper bound: 0.0018896
IS_A1_B2_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019804, upper bound: 0.0018412
IS_A1_B2_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019985, upper bound: 0.0018265
IS_A1_B2_B2_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0020271, upper bound: 0.0018446
IS_A1_B2_B2_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0020271, upper bound: 0.0018827
IS_A1_B2_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019115, upper bound: 0.0020322
IS_A1_B2_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019302, upper bound: 0.0020304
IS_A1_B2_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019009, upper bound: 0.0020175
IS_A1_B2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019213, upper bound: 0.0020213
IS_A1_B2_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0020112, upper bound: 0.0020797
IS_A1_B2_B2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0020220, upper bound: 0.0020797
IS_A1_B2_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0020044, upper bound: 0.0020730
IS_A1_B2_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0020190, upper bound: 0.0020728
IS_A2_B1_A1_B1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018511, upper bound: 0.0019736
IS_A2_B1_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018389, upper bound: 0.0019902
IS_A2_B1_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018612, upper bound: 0.0020114
IS_A2_B1_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018896, upper bound: 0.0020115
IS_A2_B1_A1_B1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018413, upper bound: 0.0019629
IS_A2_B1_A1_B1_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018263, upper bound: 0.0019762
IS_A2_B1_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018445, upper bound: 0.0020035
IS_A2_B1_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018828, upper bound: 0.0020035
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019939, upper bound: 0.0020599
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019939, upper bound: 0.0020768
IS_A2_B1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019815, upper bound: 0.0020719
IS_A2_B1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019902, upper bound: 0.0020721
IS_A2_B1_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018509, upper bound: 0.0019922
IS_A2_B1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018389, upper bound: 0.0020162
IS_A2_B1_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018612, upper bound: 0.0020348
IS_A2_B1_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018896, upper bound: 0.0020347
IS_A2_B1_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018412, upper bound: 0.0019804
IS_A2_B1_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018263, upper bound: 0.0019986
IS_A2_B1_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018446, upper bound: 0.0020271
IS_A2_B1_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018827, upper bound: 0.0020271
IS_A2_B1_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019947, upper bound: 0.0020722
IS_A2_B1_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019948, upper bound: 0.0020934
IS_A2_B1_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019897, upper bound: 0.0020719
IS_A2_B1_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019902, upper bound: 0.0020946
IS_A2_B2_A1_B1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018500, upper bound: 0.0019736
IS_A2_B2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018384, upper bound: 0.0019901
IS_A2_B2_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018610, upper bound: 0.0020115
IS_A2_B2_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018896, upper bound: 0.0020115
IS_A2_B2_A1_B1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018401, upper bound: 0.0019629
IS_A2_B2_A1_B1_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018256, upper bound: 0.0019762
IS_A2_B2_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018443, upper bound: 0.0020035
IS_A2_B2_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018828, upper bound: 0.0020034
IS_A2_B2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019795, upper bound: 0.0020724
IS_A2_B2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019891, upper bound: 0.0020719
IS_A2_B2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019814, upper bound: 0.0020719
IS_A2_B2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019903, upper bound: 0.0020720
IS_A2_B2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018501, upper bound: 0.0019922
IS_A2_B2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018384, upper bound: 0.0020162
IS_A2_B2_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018609, upper bound: 0.0020348
IS_A2_B2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018896, upper bound: 0.0020348
IS_A2_B2_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018401, upper bound: 0.0019804
IS_A2_B2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018256, upper bound: 0.0019986
IS_A2_B2_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018443, upper bound: 0.0020271
IS_A2_B2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0018828, upper bound: 0.0020271
IS_A2_B2_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019948, upper bound: 0.0020722
IS_A2_B2_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019948, upper bound: 0.0020934
IS_A2_B2_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019898, upper bound: 0.0020719
IS_A2_B2_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.87
Output dim: 5, lower bound: -0.0019903, upper bound: 0.0020945

## BFS IS instance: IS_A1_B1_A1_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0147951, -0.0063915, -0.0151396, -0.0062514, -0.0065620, 0.0066401
1: -0.0071100, -0.0047407, -0.0072071, -0.0047011, -0.0018501, 0.0018721
2: -0.0138991, 0.0035823, -0.0146156, 0.0038738, -0.0136503, 0.0138128
3: -0.0002120, 0.0021014, -0.0003068, 0.0021399, -0.0018064, 0.0018279
4: 0.0034147, 0.0164792, 0.0031969, 0.0170146, -0.0103228, 0.0102014
5: 0.9964550, 1.0000846, 0.9963944, 1.0002334, -0.0028680, 0.0028342
6: 0.0046658, 0.0079605, 0.0046109, 0.0080955, -0.0026033, 0.0025726
7: -0.0059695, 0.0063256, -0.0061745, 0.0068295, -0.0097149, 0.0096006
8: -0.0141161, -0.0045468, -0.0145083, -0.0043872, -0.0074722, 0.0075611
9: -0.0036175, -0.0027919, -0.0036312, -0.0027580, -0.0006523, 0.0006447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B2_A2_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019575, upper bound: 0.0019951
time: 1.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019687, upper bound: 0.0019949
time: 1.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0147417, -0.0061761, -0.0150833, -0.0062794, -0.0065700, 0.0067729
1: -0.0070949, -0.0046799, -0.0071912, -0.0047091, -0.0018523, 0.0019095
2: -0.0137879, 0.0040302, -0.0144985, 0.0038154, -0.0136668, 0.0140890
3: -0.0001973, 0.0021606, -0.0002914, 0.0021322, -0.0018086, 0.0018645
4: 0.0030800, 0.0163961, 0.0032405, 0.0169272, -0.0105293, 0.0102137
5: 0.9963619, 1.0000615, 0.9964065, 1.0002091, -0.0029253, 0.0028377
6: 0.0045814, 0.0079395, 0.0046219, 0.0080735, -0.0026553, 0.0025758
7: -0.0062846, 0.0062474, -0.0061335, 0.0067472, -0.0099092, 0.0096123
8: -0.0140552, -0.0043016, -0.0144442, -0.0044192, -0.0074812, 0.0077124
9: -0.0036386, -0.0027971, -0.0036285, -0.0027636, -0.0006654, 0.0006454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B2_A2_A1_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019656, upper bound: 0.0019868
time: 1.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A1_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019688, upper bound: 0.0020026
time: 1.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0148298, -0.0062553, -0.0151355, -0.0062570, -0.0066375, 0.0066693
1: -0.0071197, -0.0047023, -0.0072059, -0.0047027, -0.0018713, 0.0018803
2: -0.0139711, 0.0038655, -0.0146071, 0.0038620, -0.0138072, 0.0138734
3: -0.0002216, 0.0021388, -0.0003057, 0.0021384, -0.0018272, 0.0018359
4: 0.0032030, 0.0165330, 0.0032056, 0.0170083, -0.0103681, 0.0103187
5: 0.9963961, 1.0000995, 0.9963969, 1.0002316, -0.0028806, 0.0028668
6: 0.0046124, 0.0079740, 0.0046131, 0.0080939, -0.0026147, 0.0026022
7: -0.0061687, 0.0063762, -0.0061663, 0.0068236, -0.0097575, 0.0097110
8: -0.0141555, -0.0043917, -0.0145037, -0.0043937, -0.0075581, 0.0075943
9: -0.0036308, -0.0027885, -0.0036307, -0.0027584, -0.0006552, 0.0006521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 4.13 + 596.61 = 600.74 seconds
