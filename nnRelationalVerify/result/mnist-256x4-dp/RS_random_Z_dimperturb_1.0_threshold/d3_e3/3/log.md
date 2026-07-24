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
execution time: IAR + RelationalAnalysis = 1.49 + 2.74 = 4.23 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0023286, upper bound: 0.0023287

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022213, upper bound: 0.0022213
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022213, upper bound: 0.0022213
time: 1.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.00 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.00
Output dim: 5, lower bound: -0.0022213, upper bound: 0.0022213
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.00
Output dim: 5, lower bound: -0.0022213, upper bound: 0.0022213

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078267, 0.0078466
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022066, 0.0022123
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162812, 0.0163226
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021546, 0.0021600
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121985, 0.0121675
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033891, 0.0033805
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030763, 0.0030685
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114802, 0.0114510
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0089123, 0.0089350
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007709, 0.0007689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021965, upper bound: 0.0021962
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021962, upper bound: 0.0021965
time: 1.80 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078547, 0.0078267
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022145, 0.0022066
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0163393, 0.0162812
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021622, 0.0021546
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121675, 0.0122110
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033805, 0.0033926
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030685, 0.0030794
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114510, 0.0114919
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0089442, 0.0089123
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007689, 0.0007717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022056, upper bound: 0.0021928
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021928, upper bound: 0.0022056
time: 1.85 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.84 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.84
Output dim: 5, lower bound: -0.0021965, upper bound: 0.0021962
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.84
Output dim: 5, lower bound: -0.0021962, upper bound: 0.0021965
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.84
Output dim: 5, lower bound: -0.0022056, upper bound: 0.0021928
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.84
Output dim: 5, lower bound: -0.0021928, upper bound: 0.0022056

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077581, 0.0077729
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021873, 0.0021915
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0161385, 0.0161692
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021357, 0.0021397
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120838, 0.0120609
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033573, 0.0033509
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030474, 0.0030416
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113722, 0.0113507
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088343, 0.0088510
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007636, 0.0007622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021842, upper bound: 0.0021540
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021544, upper bound: 0.0021840
time: 1.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077530, 0.0077777
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021859, 0.0021928
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0161278, 0.0161791
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021343, 0.0021410
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120913, 0.0120529
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033593, 0.0033486
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030492, 0.0030396
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113792, 0.0113431
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088284, 0.0088565
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007641, 0.0007617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021662, upper bound: 0.0021582
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021581, upper bound: 0.0021665
time: 2.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077911, 0.0077915
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021966, 0.0021967
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162071, 0.0162078
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021448, 0.0021448
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121127, 0.0121122
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033653, 0.0033651
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030547, 0.0030545
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113994, 0.0113989
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088718, 0.0088722
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007654, 0.0007654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020637, upper bound: 0.0020497
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020612, upper bound: 0.0020514
time: 1.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078192, 0.0077634
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022045, 0.0021888
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162655, 0.0161495
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021525, 0.0021371
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120691, 0.0121558
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033532, 0.0033772
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030437, 0.0030655
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113584, 0.0114400
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0089038, 0.0088402
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007627, 0.0007682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021768, upper bound: 0.0021722
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021642, upper bound: 0.0021897
time: 1.74 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.80 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.80
Output dim: 5, lower bound: -0.0021842, upper bound: 0.0021540
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.80
Output dim: 5, lower bound: -0.0021544, upper bound: 0.0021840
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.80
Output dim: 5, lower bound: -0.0021662, upper bound: 0.0021582
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.80
Output dim: 5, lower bound: -0.0021581, upper bound: 0.0021665
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.80
Output dim: 5, lower bound: -0.0020637, upper bound: 0.0020497
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.80
Output dim: 5, lower bound: -0.0020612, upper bound: 0.0020514
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.80
Output dim: 5, lower bound: -0.0021768, upper bound: 0.0021722
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.80
Output dim: 5, lower bound: -0.0021642, upper bound: 0.0021897

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077415, 0.0077613
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021826, 0.0021882
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0161039, 0.0161451
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021311, 0.0021365
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120658, 0.0120350
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033522, 0.0033437
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030428, 0.0030351
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113553, 0.0113263
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088153, 0.0088378
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007625, 0.0007605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021745, upper bound: 0.0021258
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021583, upper bound: 0.0021438
time: 1.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077466, 0.0077562
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021841, 0.0021868
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0161145, 0.0161345
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021325, 0.0021351
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120579, 0.0120430
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033501, 0.0033459
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030408, 0.0030371
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113479, 0.0113338
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088211, 0.0088321
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007620, 0.0007610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021385, upper bound: 0.0021512
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021220, upper bound: 0.0021684
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076984, 0.0077380
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021705, 0.0021816
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160142, 0.0160966
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021192, 0.0021301
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120296, 0.0119680
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033422, 0.0033251
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030337, 0.0030182
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113212, 0.0112633
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087662, 0.0088113
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007602, 0.0007563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021376, upper bound: 0.0020621
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020663, upper bound: 0.0021294
time: 1.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077133, 0.0077274
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021747, 0.0021786
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160452, 0.0160745
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021233, 0.0021272
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120131, 0.0119912
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033376, 0.0033315
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030295, 0.0030240
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113057, 0.0112850
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087832, 0.0087992
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007592, 0.0007578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020282, upper bound: 0.0020340
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020282, upper bound: 0.0020340
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077163, 0.0077843
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021755, 0.0021947
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160514, 0.0161929
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021241, 0.0021429
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121015, 0.0119958
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033622, 0.0033328
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030518, 0.0030252
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113889, 0.0112894
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087866, 0.0088640
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007647, 0.0007581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020477, upper bound: 0.0020239
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020352, upper bound: 0.0020328
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077836, 0.0077166
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021945, 0.0021756
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0161915, 0.0160522
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021427, 0.0021242
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119964, 0.0121005
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033330, 0.0033619
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030253, 0.0030516
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112899, 0.0113879
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088633, 0.0087870
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007581, 0.0007647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020320, upper bound: 0.0019782
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019797, upper bound: 0.0020224
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076843, 0.0076715
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021665, 0.0021629
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159848, 0.0159583
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021153, 0.0021118
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119262, 0.0119461
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033135, 0.0033190
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030076, 0.0030126
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112239, 0.0112426
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087501, 0.0087356
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007537, 0.0007549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021023, upper bound: 0.0020978
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021023, upper bound: 0.0020978
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077259, 0.0076298
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021782, 0.0021511
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160715, 0.0158715
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021268, 0.0021003
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118614, 0.0120108
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032954, 0.0033370
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029913, 0.0030290
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111629, 0.0113035
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087976, 0.0086881
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007496, 0.0007590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021544, upper bound: 0.0021635
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021405, upper bound: 0.0021806
time: 1.67 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 5, lower bound: -0.0021745, upper bound: 0.0021258
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 5, lower bound: -0.0021583, upper bound: 0.0021438
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 5, lower bound: -0.0021385, upper bound: 0.0021512
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 5, lower bound: -0.0021220, upper bound: 0.0021684
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 5, lower bound: -0.0021376, upper bound: 0.0020621
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 5, lower bound: -0.0020663, upper bound: 0.0021294
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 5, lower bound: -0.0020282, upper bound: 0.0020340
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 5, lower bound: -0.0020282, upper bound: 0.0020340
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 5, lower bound: -0.0020477, upper bound: 0.0020239
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 5, lower bound: -0.0020352, upper bound: 0.0020328
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 5, lower bound: -0.0020320, upper bound: 0.0019782
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 5, lower bound: -0.0019797, upper bound: 0.0020224
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 5, lower bound: -0.0021023, upper bound: 0.0020978
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 5, lower bound: -0.0021023, upper bound: 0.0020978
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 5, lower bound: -0.0021544, upper bound: 0.0021635
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 5, lower bound: -0.0021405, upper bound: 0.0021806

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077094, 0.0077376
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021736, 0.0021815
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160371, 0.0160958
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021223, 0.0021300
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120290, 0.0119851
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033420, 0.0033298
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030335, 0.0030225
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113206, 0.0112793
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087787, 0.0088109
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007602, 0.0007574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013665, upper bound: 0.0013580
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013665, upper bound: 0.0013580
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077178, 0.0077288
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021759, 0.0021790
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160546, 0.0160775
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021246, 0.0021276
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120153, 0.0119982
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033382, 0.0033335
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030301, 0.0030258
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113077, 0.0112916
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087883, 0.0088008
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007593, 0.0007582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020821, upper bound: 0.0020690
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020821, upper bound: 0.0020690
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075960, 0.0076482
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021416, 0.0021563
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158013, 0.0159098
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020910, 0.0021054
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118900, 0.0118089
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033034, 0.0032809
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029985, 0.0029780
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111898, 0.0111135
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086496, 0.0087090
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007514, 0.0007462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021268, upper bound: 0.0021396
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021253, upper bound: 0.0021398
time: 1.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076347, 0.0076057
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021525, 0.0021443
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158818, 0.0158213
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021017, 0.0020937
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118239, 0.0118690
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032850, 0.0032976
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029818, 0.0029932
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111276, 0.0111701
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086937, 0.0086606
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007472, 0.0007500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019910, upper bound: 0.0020274
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019904, upper bound: 0.0020281
time: 1.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076690, 0.0078681
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021622, 0.0022183
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159530, 0.0163672
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021111, 0.0021659
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0122319, 0.0119223
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033984, 0.0033124
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030847, 0.0030066
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0115115, 0.0112202
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087327, 0.0089595
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007730, 0.0007534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018043, upper bound: 0.0017659
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018043, upper bound: 0.0017659
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078099, 0.0077085
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022019, 0.0021733
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162461, 0.0160353
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021499, 0.0021220
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119838, 0.0121413
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033295, 0.0033732
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030221, 0.0030619
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112781, 0.0114264
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088932, 0.0087778
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007573, 0.0007673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019857, upper bound: 0.0020412
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019857, upper bound: 0.0020413
time: 1.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076797, 0.0077010
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021652, 0.0021712
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159753, 0.0160197
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021141, 0.0021200
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119721, 0.0119389
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033262, 0.0033170
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030192, 0.0030108
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112671, 0.0112359
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087449, 0.0087692
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007566, 0.0007545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019384, upper bound: 0.0019324
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019279, upper bound: 0.0019479
time: 1.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077133, 0.0076938
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021747, 0.0021692
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160452, 0.0160046
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021233, 0.0021180
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119609, 0.0119912
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033231, 0.0033315
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030164, 0.0030240
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112565, 0.0112850
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087832, 0.0087610
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007559, 0.0007578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018936, upper bound: 0.0018964
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018936, upper bound: 0.0018964
time: 1.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075829, 0.0076979
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021379, 0.0021703
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157739, 0.0160132
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020874, 0.0021191
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119673, 0.0117885
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033249, 0.0032752
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030180, 0.0029729
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112625, 0.0110943
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086347, 0.0087656
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007563, 0.0007450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020182, upper bound: 0.0019335
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019668, upper bound: 0.0019945
time: 1.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076246, 0.0076517
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021497, 0.0021573
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158607, 0.0159171
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020989, 0.0021064
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118955, 0.0118533
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033049, 0.0032932
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029999, 0.0029892
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111950, 0.0111552
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086821, 0.0087131
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007517, 0.0007491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020246, upper bound: 0.0020217
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020243, upper bound: 0.0020222
time: 1.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078245, 0.0079063
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022060, 0.0022291
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162766, 0.0164468
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021539, 0.0021765
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0122913, 0.0121641
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0034149, 0.0033796
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030997, 0.0030676
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0115675, 0.0114478
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0089098, 0.0090030
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007767, 0.0007687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019997, upper bound: 0.0019331
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019821, upper bound: 0.0019431
time: 1.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0079639, 0.0077593
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022453, 0.0021876
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0165666, 0.0161410
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021923, 0.0021360
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120627, 0.0123809
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033514, 0.0034398
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030420, 0.0031223
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113524, 0.0116518
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0090686, 0.0088356
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007623, 0.0007824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019006, upper bound: 0.0019376
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019006, upper bound: 0.0019375
time: 1.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076390, 0.0076293
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021537, 0.0021510
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158906, 0.0158705
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021029, 0.0021002
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118606, 0.0118757
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032952, 0.0032994
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029911, 0.0029949
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111622, 0.0111763
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086985, 0.0086875
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007495, 0.0007505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020698, upper bound: 0.0020508
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020592, upper bound: 0.0020665
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076421, 0.0076276
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021546, 0.0021505
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158971, 0.0158669
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021037, 0.0020997
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118579, 0.0118805
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032945, 0.0033008
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029904, 0.0029961
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111596, 0.0111809
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087021, 0.0086855
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007493, 0.0007508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020653, upper bound: 0.0020488
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020486, upper bound: 0.0020627
time: 1.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076920, 0.0076047
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021687, 0.0021441
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160009, 0.0158194
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021175, 0.0020934
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118224, 0.0119581
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032846, 0.0033223
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029814, 0.0030156
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111262, 0.0112539
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087589, 0.0086596
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007471, 0.0007557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013747, upper bound: 0.0013730
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013747, upper bound: 0.0013730
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077009, 0.0075955
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021712, 0.0021415
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160193, 0.0158002
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021199, 0.0020909
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118081, 0.0119719
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032806, 0.0033261
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029778, 0.0030191
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111127, 0.0112669
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087690, 0.0086490
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007462, 0.0007565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013717, upper bound: 0.0013764
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013717, upper bound: 0.0013764
time: 0.80 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0013665, upper bound: 0.0013580
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0013665, upper bound: 0.0013580
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0020821, upper bound: 0.0020690
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0020821, upper bound: 0.0020690
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0021268, upper bound: 0.0021396
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0021253, upper bound: 0.0021398
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0019910, upper bound: 0.0020274
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0019904, upper bound: 0.0020281
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0018043, upper bound: 0.0017659
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0018043, upper bound: 0.0017659
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0019857, upper bound: 0.0020412
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0019857, upper bound: 0.0020413
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0019384, upper bound: 0.0019324
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0019279, upper bound: 0.0019479
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0018936, upper bound: 0.0018964
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0018936, upper bound: 0.0018964
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0020182, upper bound: 0.0019335
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0019668, upper bound: 0.0019945
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0020246, upper bound: 0.0020217
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0020243, upper bound: 0.0020222
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0019997, upper bound: 0.0019331
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0019821, upper bound: 0.0019431
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0019006, upper bound: 0.0019376
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0019006, upper bound: 0.0019375
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0020698, upper bound: 0.0020508
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0020592, upper bound: 0.0020665
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0020653, upper bound: 0.0020488
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0020486, upper bound: 0.0020627
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0013747, upper bound: 0.0013730
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0013747, upper bound: 0.0013730
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0013717, upper bound: 0.0013764
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0013717, upper bound: 0.0013764

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076759, 0.0076900
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021641, 0.0021681
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159674, 0.0159968
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021130, 0.0021169
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119550, 0.0119330
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033215, 0.0033153
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030149, 0.0030093
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112510, 0.0112303
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087406, 0.0087567
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007555, 0.0007541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018896, upper bound: 0.0018810
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018896, upper bound: 0.0018810
time: 1.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076790, 0.0076879
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021650, 0.0021675
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159739, 0.0159924
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021139, 0.0021163
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119517, 0.0119379
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033205, 0.0033167
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030140, 0.0030106
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112479, 0.0112349
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087441, 0.0087543
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007553, 0.0007544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014742, upper bound: 0.0014663
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014742, upper bound: 0.0014663
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0073635, 0.0074562
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020760, 0.0021022
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0153175, 0.0155105
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020270, 0.0020526
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115916, 0.0114473
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032205, 0.0031804
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029232, 0.0028869
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109090, 0.0107732
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0083848, 0.0084905
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007325, 0.0007234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019895, upper bound: 0.0020096
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019885, upper bound: 0.0020100
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074041, 0.0074230
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020875, 0.0020928
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154020, 0.0154414
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020382, 0.0020434
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115399, 0.0115105
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032061, 0.0031980
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029102, 0.0029028
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108603, 0.0108326
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084311, 0.0084526
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007293, 0.0007274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021020, upper bound: 0.0020991
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020774, upper bound: 0.0021160
time: 2.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075542, 0.0075634
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021298, 0.0021324
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157143, 0.0157335
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020795, 0.0020821
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117582, 0.0117439
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032668, 0.0032628
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029653, 0.0029616
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110658, 0.0110523
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086020, 0.0086125
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007430, 0.0007421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018403, upper bound: 0.0018661
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018403, upper bound: 0.0018661
time: 1.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076347, 0.0075251
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021525, 0.0021216
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158818, 0.0156538
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021017, 0.0020715
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116987, 0.0118690
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032502, 0.0032976
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029502, 0.0029932
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110098, 0.0111701
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086937, 0.0085689
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007393, 0.0007500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019802, upper bound: 0.0020177
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019803, upper bound: 0.0020178
time: 1.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077251, 0.0076620
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021780, 0.0021602
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160698, 0.0159386
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021266, 0.0021092
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119115, 0.0120096
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033094, 0.0033366
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030039, 0.0030286
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112101, 0.0113023
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087966, 0.0087248
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007527, 0.0007589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018636, upper bound: 0.0019024
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018625, upper bound: 0.0019037
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078099, 0.0076238
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022019, 0.0021494
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162461, 0.0158590
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021499, 0.0020987
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118520, 0.0121413
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032928, 0.0033732
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029889, 0.0030619
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111541, 0.0114264
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088932, 0.0086812
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007490, 0.0007673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019730, upper bound: 0.0020187
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019631, upper bound: 0.0020296
time: 2.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076012, 0.0078549
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021431, 0.0022146
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158120, 0.0163397
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020925, 0.0021623
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0122113, 0.0118169
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033927, 0.0032831
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030795, 0.0029801
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114922, 0.0111210
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086555, 0.0089444
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007717, 0.0007468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020054, upper bound: 0.0019021
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019837, upper bound: 0.0019211
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077443, 0.0077181
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021834, 0.0021760
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0161098, 0.0160553
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021319, 0.0021247
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119987, 0.0120394
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033336, 0.0033449
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030259, 0.0030362
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112922, 0.0113305
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088185, 0.0087887
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007582, 0.0007608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019432, upper bound: 0.0019449
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019111, upper bound: 0.0019702
time: 1.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074719, 0.0075282
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021066, 0.0021225
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0155431, 0.0156602
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020569, 0.0020724
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117035, 0.0116159
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032516, 0.0032273
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029514, 0.0029294
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110143, 0.0109319
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085083, 0.0085724
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007396, 0.0007341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019380, upper bound: 0.0019369
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019380, upper bound: 0.0019369
time: 1.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075017, 0.0074990
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021150, 0.0021143
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156051, 0.0155995
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020651, 0.0020643
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116581, 0.0116623
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032390, 0.0032401
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029400, 0.0029411
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109716, 0.0109755
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085422, 0.0085392
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007367, 0.0007370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020004, upper bound: 0.0019689
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019789, upper bound: 0.0019984
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076836, 0.0077995
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021663, 0.0021990
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159835, 0.0162246
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021152, 0.0021471
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121252, 0.0119451
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033687, 0.0033187
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030578, 0.0030124
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114112, 0.0112417
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087494, 0.0088813
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007662, 0.0007549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019793, upper bound: 0.0019085
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019681, upper bound: 0.0019115
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077095, 0.0077668
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021736, 0.0021898
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160373, 0.0161565
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021223, 0.0021381
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120744, 0.0119853
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033546, 0.0033299
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030450, 0.0030225
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113633, 0.0112795
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087788, 0.0088441
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007630, 0.0007574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019606, upper bound: 0.0019168
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019534, upper bound: 0.0019218
time: 2.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075446, 0.0075760
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021271, 0.0021360
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156943, 0.0157597
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020769, 0.0020855
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117778, 0.0117289
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032722, 0.0032586
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029702, 0.0029579
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110842, 0.0110382
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085911, 0.0086269
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007443, 0.0007412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020630, upper bound: 0.0020357
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020432, upper bound: 0.0020443
time: 2.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075874, 0.0075349
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021392, 0.0021244
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157833, 0.0156741
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020887, 0.0020742
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117139, 0.0117954
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032545, 0.0032771
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029541, 0.0029746
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110240, 0.0111008
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086398, 0.0085800
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007402, 0.0007454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019731, upper bound: 0.0019646
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019552, upper bound: 0.0019786
time: 2.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074903, 0.0075037
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021118, 0.0021156
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0155813, 0.0156092
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020619, 0.0020656
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116653, 0.0116445
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032410, 0.0032352
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029418, 0.0029366
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109784, 0.0109588
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085292, 0.0085445
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007372, 0.0007359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020398, upper bound: 0.0020231
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020397, upper bound: 0.0020237
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075226, 0.0074759
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021209, 0.0021077
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156485, 0.0155513
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020708, 0.0020580
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116221, 0.0116947
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032290, 0.0032491
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029309, 0.0029492
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109377, 0.0110060
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085660, 0.0085128
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007344, 0.0007390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018707, upper bound: 0.0018853
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018707, upper bound: 0.0018853
time: 1.78 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0018896, upper bound: 0.0018810
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0018896, upper bound: 0.0018810
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0014742, upper bound: 0.0014663
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0014742, upper bound: 0.0014663
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0019895, upper bound: 0.0020096
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0019885, upper bound: 0.0020100
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0021020, upper bound: 0.0020991
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0020774, upper bound: 0.0021160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0018403, upper bound: 0.0018661
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0018403, upper bound: 0.0018661
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0019802, upper bound: 0.0020177
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0019803, upper bound: 0.0020178
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0018636, upper bound: 0.0019024
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0018625, upper bound: 0.0019037
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0019730, upper bound: 0.0020187
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0019631, upper bound: 0.0020296
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0020054, upper bound: 0.0019021
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0019837, upper bound: 0.0019211
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0019432, upper bound: 0.0019449
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0019111, upper bound: 0.0019702
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0019380, upper bound: 0.0019369
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0019380, upper bound: 0.0019369
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0020004, upper bound: 0.0019689
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0019789, upper bound: 0.0019984
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0019793, upper bound: 0.0019085
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0019681, upper bound: 0.0019115
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0019606, upper bound: 0.0019168
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0019534, upper bound: 0.0019218
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0020630, upper bound: 0.0020357
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0020432, upper bound: 0.0020443
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0019731, upper bound: 0.0019646
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0019552, upper bound: 0.0019786
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0020398, upper bound: 0.0020231
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0020397, upper bound: 0.0020237
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0018707, upper bound: 0.0018853
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.95
Output dim: 5, lower bound: -0.0018707, upper bound: 0.0018853

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072713, 0.0074031
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020501, 0.0020872
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0151259, 0.0153999
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020017, 0.0020379
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115090, 0.0113041
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031975, 0.0031406
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029024, 0.0028507
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108312, 0.0106384
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082799, 0.0084299
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007273, 0.0007144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019688, upper bound: 0.0019732
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019549, upper bound: 0.0019896
time: 1.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0073635, 0.0073641
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020760, 0.0020762
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0153175, 0.0153188
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020270, 0.0020272
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0114483, 0.0114473
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031807, 0.0031804
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028871, 0.0028869
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0107742, 0.0107732
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0083848, 0.0083856
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007235, 0.0007234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019582, upper bound: 0.0018992
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018997, upper bound: 0.0019808
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0071717, 0.0072498
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020220, 0.0020440
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0149186, 0.0150810
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019742, 0.0019957
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0112706, 0.0111492
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031313, 0.0030976
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028423, 0.0028117
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106069, 0.0104927
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0081665, 0.0082554
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007122, 0.0007046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020293, upper bound: 0.0020264
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020293, upper bound: 0.0020264
time: 1.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072074, 0.0071906
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020320, 0.0020273
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0149929, 0.0149580
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019841, 0.0019795
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0111787, 0.0112047
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031058, 0.0031130
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028191, 0.0028257
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0105204, 0.0105449
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082071, 0.0081880
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007064, 0.0007081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019411, upper bound: 0.0019821
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019411, upper bound: 0.0019821
time: 1.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074017, 0.0073216
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020868, 0.0020642
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0153971, 0.0152304
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020376, 0.0020155
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113822, 0.0115068
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031623, 0.0031969
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028704, 0.0029019
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0107119, 0.0108292
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084284, 0.0083371
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007193, 0.0007272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019507, upper bound: 0.0019256
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018711, upper bound: 0.0019880
time: 1.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074428, 0.0072918
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020984, 0.0020558
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154825, 0.0151685
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020489, 0.0020073
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113360, 0.0115706
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031495, 0.0032147
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028588, 0.0029179
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106684, 0.0108892
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084751, 0.0083032
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007164, 0.0007312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019727, upper bound: 0.0020100
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019707, upper bound: 0.0020103
time: 2.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077776, 0.0075991
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021928, 0.0021425
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0161790, 0.0158078
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021410, 0.0020919
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118137, 0.0120912
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032822, 0.0033593
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029793, 0.0030492
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111180, 0.0113791
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088564, 0.0086532
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007466, 0.0007641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019401, upper bound: 0.0019677
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019269, upper bound: 0.0019858
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077853, 0.0075907
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021950, 0.0021401
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0161949, 0.0157903
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021431, 0.0020896
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118007, 0.0121031
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032786, 0.0033626
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029760, 0.0030522
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111057, 0.0113903
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088651, 0.0086436
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007457, 0.0007648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019486, upper bound: 0.0020135
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019462, upper bound: 0.0020152
time: 2.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075870, 0.0078454
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021391, 0.0022119
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157825, 0.0163200
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020886, 0.0021597
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121965, 0.0117949
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033886, 0.0032770
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030758, 0.0029745
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114783, 0.0111003
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086394, 0.0089336
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007707, 0.0007454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019769, upper bound: 0.0018709
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019698, upper bound: 0.0018742
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075921, 0.0078407
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021405, 0.0022106
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157932, 0.0163103
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020900, 0.0021584
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121893, 0.0118028
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033866, 0.0032792
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030740, 0.0029765
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114715, 0.0111078
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086452, 0.0089283
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007703, 0.0007459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019590, upper bound: 0.0018969
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019566, upper bound: 0.0018971
time: 1.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0073921, 0.0074417
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020841, 0.0020981
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0153770, 0.0154803
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020349, 0.0020486
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115690, 0.0114918
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032142, 0.0031928
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029175, 0.0028981
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108877, 0.0108151
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084174, 0.0084739
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007311, 0.0007262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019707, upper bound: 0.0018875
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019028, upper bound: 0.0019385
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074464, 0.0073906
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020994, 0.0020837
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154900, 0.0153739
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020499, 0.0020345
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0114895, 0.0115762
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031921, 0.0032162
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028975, 0.0029194
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108129, 0.0108945
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084792, 0.0084157
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007261, 0.0007315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019651, upper bound: 0.0019828
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019607, upper bound: 0.0019848
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075111, 0.0075512
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021177, 0.0021290
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156246, 0.0157080
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020677, 0.0020787
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117392, 0.0116768
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032615, 0.0032442
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029604, 0.0029447
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110479, 0.0109892
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085529, 0.0085986
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007418, 0.0007379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020515, upper bound: 0.0020244
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020497, upper bound: 0.0020242
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075197, 0.0075410
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021201, 0.0021261
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156426, 0.0156867
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020700, 0.0020759
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117233, 0.0116903
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032571, 0.0032479
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029564, 0.0029481
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110329, 0.0110018
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085628, 0.0085869
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007408, 0.0007388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020204, upper bound: 0.0020094
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019997, upper bound: 0.0020211
time: 2.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074362, 0.0074456
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020965, 0.0020992
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154689, 0.0154884
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020471, 0.0020496
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115751, 0.0115605
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032159, 0.0032118
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029191, 0.0029154
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108934, 0.0108797
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084677, 0.0084784
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007315, 0.0007305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020259, upper bound: 0.0020063
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020237, upper bound: 0.0020091
time: 1.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074339, 0.0074504
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020959, 0.0021005
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154641, 0.0154984
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020464, 0.0020510
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115825, 0.0115569
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032180, 0.0032108
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029209, 0.0029145
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109004, 0.0108763
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084651, 0.0084838
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007319, 0.0007303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020282, upper bound: 0.0020123
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020257, upper bound: 0.0020120
time: 1.64 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 6.55 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019688, upper bound: 0.0019732
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019549, upper bound: 0.0019896
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019582, upper bound: 0.0018992
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0018997, upper bound: 0.0019808
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0020293, upper bound: 0.0020264
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0020293, upper bound: 0.0020264
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019411, upper bound: 0.0019821
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019411, upper bound: 0.0019821
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019507, upper bound: 0.0019256
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0018711, upper bound: 0.0019880
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019727, upper bound: 0.0020100
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019707, upper bound: 0.0020103
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019401, upper bound: 0.0019677
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019269, upper bound: 0.0019858
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019486, upper bound: 0.0020135
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019462, upper bound: 0.0020152
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019769, upper bound: 0.0018709
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019698, upper bound: 0.0018742
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019590, upper bound: 0.0018969
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019566, upper bound: 0.0018971
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019707, upper bound: 0.0018875
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019028, upper bound: 0.0019385
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019651, upper bound: 0.0019828
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019607, upper bound: 0.0019848
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0020515, upper bound: 0.0020244
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0020497, upper bound: 0.0020242
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0020204, upper bound: 0.0020094
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0019997, upper bound: 0.0020211
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0020259, upper bound: 0.0020063
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0020237, upper bound: 0.0020091
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0020282, upper bound: 0.0020123
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.55
Output dim: 5, lower bound: -0.0020257, upper bound: 0.0020120

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072092, 0.0073043
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020325, 0.0020594
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0149966, 0.0151945
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019846, 0.0020107
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113554, 0.0112075
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031549, 0.0031138
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028637, 0.0028264
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106867, 0.0105475
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082091, 0.0083175
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007176, 0.0007082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019471, upper bound: 0.0019814
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019462, upper bound: 0.0019821
time: 1.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075219, 0.0073638
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021207, 0.0020761
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156471, 0.0153183
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020706, 0.0020271
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0114479, 0.0116937
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031806, 0.0032489
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028870, 0.0029490
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0107738, 0.0110051
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085653, 0.0083853
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007234, 0.0007390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015115, upper bound: 0.0015458
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015115, upper bound: 0.0015458
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0071279, 0.0072094
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020096, 0.0020326
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0148274, 0.0149970
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019622, 0.0019846
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0112078, 0.0110811
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031139, 0.0030787
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028264, 0.0027945
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0105478, 0.0104285
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0081165, 0.0082094
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007083, 0.0007003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019994, upper bound: 0.0019010
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019344, upper bound: 0.0019983
time: 1.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0071313, 0.0072078
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020106, 0.0020322
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0148345, 0.0149938
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019631, 0.0019842
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0112054, 0.0110864
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031132, 0.0030801
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028258, 0.0027958
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0105455, 0.0104335
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0081204, 0.0082076
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007081, 0.0007006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020154, upper bound: 0.0020094
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020107, upper bound: 0.0020122
time: 1.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0071730, 0.0071615
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020223, 0.0020191
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0149214, 0.0148975
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019746, 0.0019714
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0111334, 0.0111513
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0030932, 0.0030982
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028077, 0.0028122
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0104778, 0.0104946
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0081680, 0.0081549
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007036, 0.0007047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019254, upper bound: 0.0019572
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019166, upper bound: 0.0019663
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072074, 0.0071563
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020320, 0.0020176
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0149929, 0.0148865
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019841, 0.0019700
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0111252, 0.0112047
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0030909, 0.0031130
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028056, 0.0028257
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0104701, 0.0105449
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082071, 0.0081489
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007030, 0.0007081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019061, upper bound: 0.0019340
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018984, upper bound: 0.0019432
time: 1.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075515, 0.0073213
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021290, 0.0020642
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157086, 0.0152298
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020788, 0.0020154
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113818, 0.0117397
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031622, 0.0032616
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028703, 0.0029606
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0107116, 0.0110483
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085989, 0.0083368
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007193, 0.0007419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015040, upper bound: 0.0015484
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015040, upper bound: 0.0015484
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074250, 0.0072762
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020934, 0.0020514
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154454, 0.0151359
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020440, 0.0020030
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113117, 0.0115430
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031427, 0.0032070
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028526, 0.0029110
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106455, 0.0108632
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084549, 0.0082854
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007148, 0.0007294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019374, upper bound: 0.0019565
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019294, upper bound: 0.0019765
time: 2.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074270, 0.0072719
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020939, 0.0020502
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154496, 0.0151271
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020445, 0.0020018
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113051, 0.0115461
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031409, 0.0032078
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028510, 0.0029117
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106393, 0.0108661
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084571, 0.0082806
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007144, 0.0007296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019688, upper bound: 0.0019704
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019385, upper bound: 0.0020084
time: 2.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076809, 0.0074720
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021655, 0.0021066
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159779, 0.0155432
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021144, 0.0020569
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116160, 0.0119409
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032273, 0.0033175
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029294, 0.0030113
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109320, 0.0112377
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087463, 0.0085084
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007341, 0.0007546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019154, upper bound: 0.0019738
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019146, upper bound: 0.0019745
time: 1.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077159, 0.0075270
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021754, 0.0021221
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160506, 0.0156577
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021240, 0.0020720
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117016, 0.0119952
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032510, 0.0033326
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029510, 0.0030250
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110125, 0.0112888
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087861, 0.0085710
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007395, 0.0007580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018786, upper bound: 0.0019425
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018786, upper bound: 0.0019426
time: 1.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077227, 0.0075188
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021773, 0.0021198
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160648, 0.0156406
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021259, 0.0020698
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116888, 0.0120058
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032475, 0.0033356
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029478, 0.0030277
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110005, 0.0112988
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087939, 0.0085617
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007387, 0.0007587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018094, upper bound: 0.0018718
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018094, upper bound: 0.0018720
time: 2.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074269, 0.0073708
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020939, 0.0020781
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154494, 0.0153327
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020445, 0.0020290
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0114587, 0.0115459
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031836, 0.0032078
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028897, 0.0029117
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0107839, 0.0108660
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084570, 0.0083931
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007241, 0.0007296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018097, upper bound: 0.0018228
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018097, upper bound: 0.0018228
time: 1.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074264, 0.0073709
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020938, 0.0020781
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154483, 0.0153330
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020443, 0.0020291
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0114589, 0.0115451
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031836, 0.0032076
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028898, 0.0029115
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0107841, 0.0108652
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084564, 0.0083933
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007241, 0.0007296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018693, upper bound: 0.0018784
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018593, upper bound: 0.0019031
time: 1.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0073525, 0.0074204
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020730, 0.0020921
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0152948, 0.0154359
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020240, 0.0020427
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115359, 0.0114304
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032050, 0.0031757
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029092, 0.0028826
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108565, 0.0107572
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0083724, 0.0084497
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007290, 0.0007223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018906, upper bound: 0.0018788
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018903, upper bound: 0.0018788
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0073806, 0.0073872
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020809, 0.0020827
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0153532, 0.0153669
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020318, 0.0020336
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0114842, 0.0114740
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031907, 0.0031878
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028962, 0.0028936
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108079, 0.0107983
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084044, 0.0084118
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007257, 0.0007251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020248, upper bound: 0.0019996
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020241, upper bound: 0.0019994
time: 2.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074162, 0.0074935
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020909, 0.0021127
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154272, 0.0155881
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020415, 0.0020628
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116496, 0.0115293
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032366, 0.0032032
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029378, 0.0029075
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109635, 0.0108504
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084449, 0.0085329
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007362, 0.0007286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019958, upper bound: 0.0019850
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019954, upper bound: 0.0019842
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074667, 0.0074387
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021051, 0.0020973
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0155322, 0.0154741
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020554, 0.0020477
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115643, 0.0116078
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032129, 0.0032250
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029164, 0.0029273
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108833, 0.0109242
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085023, 0.0084705
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007308, 0.0007335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019852, upper bound: 0.0020066
time: 2.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019837, upper bound: 0.0020070
time: 1.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074105, 0.0074195
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020893, 0.0020918
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154154, 0.0154342
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020400, 0.0020425
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115345, 0.0115205
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032046, 0.0032007
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029088, 0.0029053
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108553, 0.0108421
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084384, 0.0084487
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007289, 0.0007280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020241, upper bound: 0.0019690
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019886, upper bound: 0.0020045
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074101, 0.0074198
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020892, 0.0020919
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154145, 0.0154346
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020399, 0.0020425
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115349, 0.0115198
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032047, 0.0032005
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029089, 0.0029051
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108556, 0.0108414
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084379, 0.0084489
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007289, 0.0007280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020168, upper bound: 0.0019994
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020168, upper bound: 0.0020019
time: 1.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072103, 0.0072532
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020329, 0.0020449
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0149990, 0.0150881
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019849, 0.0019967
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0112759, 0.0112093
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031328, 0.0031143
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028436, 0.0028268
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106119, 0.0105492
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082104, 0.0082592
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007126, 0.0007084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020156, upper bound: 0.0019787
time: 2.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019875, upper bound: 0.0019989
time: 2.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072382, 0.0072127
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020407, 0.0020335
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0150569, 0.0150038
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019925, 0.0019855
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0112129, 0.0112526
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031153, 0.0031263
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028277, 0.0028377
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0105526, 0.0105899
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082422, 0.0082131
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007086, 0.0007111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0019953
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020093, upper bound: 0.0019979
time: 1.76 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 5.09 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019471, upper bound: 0.0019814
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019462, upper bound: 0.0019821
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0015115, upper bound: 0.0015458
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0015115, upper bound: 0.0015458
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019994, upper bound: 0.0019010
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019344, upper bound: 0.0019983
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0020154, upper bound: 0.0020094
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0020107, upper bound: 0.0020122
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019254, upper bound: 0.0019572
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019166, upper bound: 0.0019663
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019061, upper bound: 0.0019340
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0018984, upper bound: 0.0019432
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0015040, upper bound: 0.0015484
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0015040, upper bound: 0.0015484
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019374, upper bound: 0.0019565
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019294, upper bound: 0.0019765
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019688, upper bound: 0.0019704
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019385, upper bound: 0.0020084
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019154, upper bound: 0.0019738
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019146, upper bound: 0.0019745
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0018786, upper bound: 0.0019425
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0018786, upper bound: 0.0019426
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0018094, upper bound: 0.0018718
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0018094, upper bound: 0.0018720
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0018097, upper bound: 0.0018228
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0018097, upper bound: 0.0018228
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0018693, upper bound: 0.0018784
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0018593, upper bound: 0.0019031
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0018906, upper bound: 0.0018788
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0018903, upper bound: 0.0018788
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0020248, upper bound: 0.0019996
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0020241, upper bound: 0.0019994
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019958, upper bound: 0.0019850
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019954, upper bound: 0.0019842
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019852, upper bound: 0.0020066
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019837, upper bound: 0.0020070
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0020241, upper bound: 0.0019690
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019886, upper bound: 0.0020045
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0020168, upper bound: 0.0019994
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0020168, upper bound: 0.0020019
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0020156, upper bound: 0.0019787
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0019875, upper bound: 0.0019989
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0019953
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.09
Output dim: 5, lower bound: -0.0020093, upper bound: 0.0019979

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0071893, 0.0072864
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020269, 0.0020543
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0149551, 0.0151573
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019791, 0.0020058
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113276, 0.0111765
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031471, 0.0031052
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028567, 0.0028186
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106605, 0.0105184
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0081865, 0.0082971
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007158, 0.0007063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018647, upper bound: 0.0018794
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018405, upper bound: 0.0018959
time: 1.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0071913, 0.0072823
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020275, 0.0020532
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0149594, 0.0151488
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019796, 0.0020047
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113212, 0.0111797
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031454, 0.0031061
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028551, 0.0028194
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106546, 0.0105213
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0081888, 0.0082925
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007154, 0.0007065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019416, upper bound: 0.0019664
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019330, upper bound: 0.0019773
time: 2.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0070793, 0.0072991
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0019959, 0.0020579
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0147264, 0.0151835
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019488, 0.0020093
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113472, 0.0110056
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031526, 0.0030577
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028616, 0.0027755
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106790, 0.0103575
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0080613, 0.0083115
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007171, 0.0006955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019850, upper bound: 0.0018824
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019824, upper bound: 0.0018863
time: 1.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072384, 0.0071608
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020408, 0.0020189
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0150574, 0.0148960
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019926, 0.0019712
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0111323, 0.0112530
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0030929, 0.0031264
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028074, 0.0028378
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0104767, 0.0105903
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082424, 0.0081541
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007035, 0.0007111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018545, upper bound: 0.0018996
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018291, upper bound: 0.0019077
time: 2.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0071138, 0.0071897
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020056, 0.0020271
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0147981, 0.0149561
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019583, 0.0019792
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0111772, 0.0110591
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031054, 0.0030726
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028187, 0.0027890
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0105190, 0.0104079
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0081005, 0.0081870
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007063, 0.0006989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019856, upper bound: 0.0019748
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019754, upper bound: 0.0019809
time: 1.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0071134, 0.0071903
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020055, 0.0020272
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0147974, 0.0149573
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019582, 0.0019794
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0111781, 0.0110586
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031056, 0.0030724
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028190, 0.0027888
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0105199, 0.0104074
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0081001, 0.0081876
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007064, 0.0006988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019302, upper bound: 0.0019324
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019302, upper bound: 0.0019324
time: 1.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074421, 0.0072653
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020982, 0.0020484
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154811, 0.0151134
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020487, 0.0020000
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0112948, 0.0115696
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031380, 0.0032144
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028484, 0.0029177
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106297, 0.0108883
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084744, 0.0082731
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007138, 0.0007311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017981, upper bound: 0.0018471
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017981, upper bound: 0.0018471
time: 1.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072500, 0.0072541
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020441, 0.0020452
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0150816, 0.0150900
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019958, 0.0019969
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0112773, 0.0112710
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031332, 0.0031314
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028440, 0.0028424
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106132, 0.0106073
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082557, 0.0082603
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007127, 0.0007123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020106, upper bound: 0.0019848
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020098, upper bound: 0.0019849
time: 1.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072497, 0.0072440
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020439, 0.0020424
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0150808, 0.0150691
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019957, 0.0019941
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0112617, 0.0112704
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031288, 0.0031313
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028400, 0.0028422
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0105985, 0.0106067
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082552, 0.0082488
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007117, 0.0007122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019948, upper bound: 0.0018926
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019369, upper bound: 0.0019703
time: 2.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072084, 0.0072830
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020323, 0.0020533
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0149950, 0.0151501
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019844, 0.0020049
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113222, 0.0112063
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031456, 0.0031135
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028553, 0.0028261
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106555, 0.0105464
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082083, 0.0082932
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007155, 0.0007082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019210, upper bound: 0.0019071
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019210, upper bound: 0.0019071
time: 2.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072073, 0.0072658
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020320, 0.0020485
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0149927, 0.0151143
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019840, 0.0020001
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0112955, 0.0112046
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031382, 0.0031130
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028485, 0.0028256
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106303, 0.0105448
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082070, 0.0082736
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007138, 0.0007081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013838, upper bound: 0.0013813
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013838, upper bound: 0.0013813
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074006, 0.0073788
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020865, 0.0020804
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0153947, 0.0153494
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020372, 0.0020313
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0114712, 0.0115051
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031870, 0.0031964
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028929, 0.0029014
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0107957, 0.0108275
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084271, 0.0084023
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007249, 0.0007270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019109, upper bound: 0.0019263
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019109, upper bound: 0.0019263
time: 1.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074057, 0.0073691
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020879, 0.0020776
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154053, 0.0153293
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020386, 0.0020286
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0114562, 0.0115129
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031829, 0.0031986
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028891, 0.0029034
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0107815, 0.0108350
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084329, 0.0083913
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007240, 0.0007275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017860, upper bound: 0.0018134
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017860, upper bound: 0.0018134
time: 1.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074201, 0.0074516
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020920, 0.0021009
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154353, 0.0155008
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020426, 0.0020513
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115843, 0.0115354
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032185, 0.0032049
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029214, 0.0029091
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109021, 0.0108561
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084493, 0.0084851
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007321, 0.0007290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018388, upper bound: 0.0018017
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018388, upper bound: 0.0018017
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074430, 0.0074293
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020984, 0.0020946
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154829, 0.0154545
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020489, 0.0020452
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115497, 0.0115709
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032089, 0.0032148
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029127, 0.0029180
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108696, 0.0108895
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084754, 0.0084598
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007299, 0.0007312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019580, upper bound: 0.0018877
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019230, upper bound: 0.0019764
time: 2.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0073918, 0.0074036
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020840, 0.0020874
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0153765, 0.0154010
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020348, 0.0020381
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115098, 0.0114915
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031978, 0.0031927
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029026, 0.0028980
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108320, 0.0108147
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084171, 0.0084305
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007273, 0.0007262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020053, upper bound: 0.0019880
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020031, upper bound: 0.0019878
time: 1.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0073941, 0.0074004
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020847, 0.0020865
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0153813, 0.0153944
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020355, 0.0020372
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115048, 0.0114950
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031964, 0.0031937
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029013, 0.0028989
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108273, 0.0108181
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084197, 0.0084269
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007270, 0.0007264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018500, upper bound: 0.0018388
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018499, upper bound: 0.0018388
time: 1.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0071936, 0.0072416
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020281, 0.0020417
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0149642, 0.0150640
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019803, 0.0019935
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0112579, 0.0111833
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031278, 0.0031070
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028391, 0.0028203
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0105950, 0.0105247
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0081914, 0.0082461
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007114, 0.0007067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020014, upper bound: 0.0019627
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020003, upper bound: 0.0019641
time: 1.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0071993, 0.0072364
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020298, 0.0020402
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0149760, 0.0150532
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019818, 0.0019921
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0112499, 0.0111921
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031255, 0.0031095
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028371, 0.0028225
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0105874, 0.0105331
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0081979, 0.0082402
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007109, 0.0007073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018182, upper bound: 0.0018371
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018181, upper bound: 0.0018371
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072236, 0.0071974
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020366, 0.0020292
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0150266, 0.0149721
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019885, 0.0019813
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0111892, 0.0112299
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031087, 0.0031200
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028218, 0.0028320
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0105303, 0.0105686
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082256, 0.0081958
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007071, 0.0007097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019238, upper bound: 0.0018938
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019103, upper bound: 0.0019097
time: 1.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072227, 0.0071979
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020363, 0.0020294
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0150246, 0.0149732
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019883, 0.0019815
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0111900, 0.0112285
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031089, 0.0031196
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028220, 0.0028317
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0105310, 0.0105672
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082245, 0.0081963
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007071, 0.0007096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018264, upper bound: 0.0018217
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018264, upper bound: 0.0018217
time: 1.76 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 4.94 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0018647, upper bound: 0.0018794
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0018405, upper bound: 0.0018959
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0019416, upper bound: 0.0019664
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0019330, upper bound: 0.0019773
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0019850, upper bound: 0.0018824
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0019824, upper bound: 0.0018863
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0018545, upper bound: 0.0018996
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0018291, upper bound: 0.0019077
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0019856, upper bound: 0.0019748
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0019754, upper bound: 0.0019809
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0019302, upper bound: 0.0019324
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0019302, upper bound: 0.0019324
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0017981, upper bound: 0.0018471
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0017981, upper bound: 0.0018471
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0020106, upper bound: 0.0019848
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0020098, upper bound: 0.0019849
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0019948, upper bound: 0.0018926
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0019369, upper bound: 0.0019703
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0019210, upper bound: 0.0019071
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0019210, upper bound: 0.0019071
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0013838, upper bound: 0.0013813
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0013838, upper bound: 0.0013813
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0019109, upper bound: 0.0019263
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0019109, upper bound: 0.0019263
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0017860, upper bound: 0.0018134
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0017860, upper bound: 0.0018134
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0018388, upper bound: 0.0018017
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0018388, upper bound: 0.0018017
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0019580, upper bound: 0.0018877
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0019230, upper bound: 0.0019764
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0020053, upper bound: 0.0019880
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0020031, upper bound: 0.0019878
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0018500, upper bound: 0.0018388
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0018499, upper bound: 0.0018388
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0020014, upper bound: 0.0019627
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0020003, upper bound: 0.0019641
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0018182, upper bound: 0.0018371
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0018181, upper bound: 0.0018371
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0019238, upper bound: 0.0018938
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0019103, upper bound: 0.0019097
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0018264, upper bound: 0.0018217
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.94
Output dim: 5, lower bound: -0.0018264, upper bound: 0.0018217

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0070763, 0.0073045
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0019951, 0.0020594
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0147201, 0.0151949
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019480, 0.0020108
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113557, 0.0110009
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031549, 0.0030564
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028637, 0.0027743
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106870, 0.0103530
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0080578, 0.0083177
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007176, 0.0006952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019714, upper bound: 0.0018620
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019575, upper bound: 0.0018684
time: 2.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0070848, 0.0073030
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0019975, 0.0020590
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0147377, 0.0151917
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019503, 0.0020104
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113533, 0.0110141
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031543, 0.0030600
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028631, 0.0027776
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106847, 0.0103655
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0080675, 0.0083159
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007175, 0.0006960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018996, upper bound: 0.0017852
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018772, upper bound: 0.0017997
time: 2.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0070262, 0.0071124
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0019809, 0.0020053
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0146159, 0.0147953
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019342, 0.0019579
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0110571, 0.0109230
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0030720, 0.0030347
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0027884, 0.0027546
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0104060, 0.0102797
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0080007, 0.0080990
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0006987, 0.0006903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018070, upper bound: 0.0017952
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018070, upper bound: 0.0017952
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0070365, 0.0070910
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0019838, 0.0019992
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0146373, 0.0147508
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019370, 0.0019520
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0110238, 0.0109390
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0030627, 0.0030392
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0027800, 0.0027587
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0103746, 0.0102948
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0080125, 0.0080746
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0006966, 0.0006913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019610, upper bound: 0.0019642
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019593, upper bound: 0.0019662
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072341, 0.0072385
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020396, 0.0020408
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0150483, 0.0150576
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019914, 0.0019926
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0112531, 0.0112462
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031265, 0.0031245
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028379, 0.0028361
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0105904, 0.0105839
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082375, 0.0082426
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007111, 0.0007107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013849, upper bound: 0.0013891
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013849, upper bound: 0.0013891
time: 1.05 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 3.52 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.52
Output dim: 5, lower bound: -0.0019714, upper bound: 0.0018620
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.52
Output dim: 5, lower bound: -0.0019575, upper bound: 0.0018684
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.52
Output dim: 5, lower bound: -0.0018996, upper bound: 0.0017852
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.52
Output dim: 5, lower bound: -0.0018772, upper bound: 0.0017997
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.52
Output dim: 5, lower bound: -0.0018070, upper bound: 0.0017952
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.52
Output dim: 5, lower bound: -0.0018070, upper bound: 0.0017952
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.52
Output dim: 5, lower bound: -0.0019610, upper bound: 0.0019642
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.52
Output dim: 5, lower bound: -0.0019593, upper bound: 0.0019662
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.52
Output dim: 5, lower bound: -0.0013849, upper bound: 0.0013891
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.52
Output dim: 5, lower bound: -0.0013849, upper bound: 0.0013891
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.52
Output dim: 5, lower bound: -0.0020098, upper bound: 0.0019849
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.52
Output dim: 5, lower bound: -0.0019948, upper bound: 0.0018926
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.52
Output dim: 5, lower bound: -0.0020053, upper bound: 0.0019880
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.52
Output dim: 5, lower bound: -0.0020031, upper bound: 0.0019878
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.52
Output dim: 5, lower bound: -0.0020014, upper bound: 0.0019627
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.52
Output dim: 5, lower bound: -0.0020003, upper bound: 0.0019641

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 4.23 + 597.03 = 601.26 seconds
