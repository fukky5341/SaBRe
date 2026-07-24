## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 27.1733048946


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570)
1: (-16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204)
2: (-27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886)
3: (-24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909)
4: (-24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434)
5: (-18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721)
6: (-19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750)
7: (-22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555)
8: (-25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507)
9: (-17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.39 + 6.40 = 7.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -27.2005054, upper bound: 27.2005054

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1925094, upper bound: 27.1925094
time: 3.40 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1925094, upper bound: 27.1925094
time: 3.16 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.57
Output dim: 2, lower bound: -27.1925094, upper bound: 27.1925094
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.57
Output dim: 2, lower bound: -27.1925094, upper bound: 27.1925094

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1751285, upper bound: 27.1751285
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1751285, upper bound: 27.1751285
time: 4.70 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1769955, upper bound: 27.1769955
time: 3.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1769955, upper bound: 27.1769955
time: 3.23 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 9.77 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 9.77
Output dim: 2, lower bound: -27.1751285, upper bound: 27.1751285
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 9.77
Output dim: 2, lower bound: -27.1751285, upper bound: 27.1751285
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 9.77
Output dim: 2, lower bound: -27.1769955, upper bound: 27.1769955
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 9.77
Output dim: 2, lower bound: -27.1769955, upper bound: 27.1769955

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1751285, upper bound: 27.1751083
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1751083, upper bound: 27.1751285
time: 14.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1750863, upper bound: 27.1750863
time: 3.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1750863, upper bound: 27.1750863
time: 3.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1490694, upper bound: 27.1490694
time: 3.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1490694, upper bound: 27.1490694
time: 2.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1769955, upper bound: 27.1769930
time: 5.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1769931, upper bound: 27.1769954
time: 3.10 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 10.17 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 10.17
Output dim: 2, lower bound: -27.1751285, upper bound: 27.1751083
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 10.17
Output dim: 2, lower bound: -27.1751083, upper bound: 27.1751285
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 10.17
Output dim: 2, lower bound: -27.1750863, upper bound: 27.1750863
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 10.17
Output dim: 2, lower bound: -27.1750863, upper bound: 27.1750863
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 10.17
Output dim: 2, lower bound: -27.1490694, upper bound: 27.1490694
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 10.17
Output dim: 2, lower bound: -27.1490694, upper bound: 27.1490694
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 10.17
Output dim: 2, lower bound: -27.1769955, upper bound: 27.1769930
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 10.17
Output dim: 2, lower bound: -27.1769931, upper bound: 27.1769954

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1740302, upper bound: 27.1740217
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1740302, upper bound: 27.1740217
time: 9.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1751069, upper bound: 27.1751285
time: 8.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1751083, upper bound: 27.1751263
time: 5.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1735549, upper bound: 27.1735493
time: 3.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1735493, upper bound: 27.1735549
time: 3.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1750854, upper bound: 27.1750854
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1750854, upper bound: 27.1750854
time: 4.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1769950, upper bound: 27.1769931
time: 3.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1769955, upper bound: 27.1769917
time: 13.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1710265, upper bound: 27.1710310
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1710265, upper bound: 27.1710310
time: 8.48 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.96
Output dim: 2, lower bound: -27.1740302, upper bound: 27.1740217
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.96
Output dim: 2, lower bound: -27.1740302, upper bound: 27.1740217
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.96
Output dim: 2, lower bound: -27.1751069, upper bound: 27.1751285
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.96
Output dim: 2, lower bound: -27.1751083, upper bound: 27.1751263
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.96
Output dim: 2, lower bound: -27.1735549, upper bound: 27.1735493
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.96
Output dim: 2, lower bound: -27.1735493, upper bound: 27.1735549
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.96
Output dim: 2, lower bound: -27.1750854, upper bound: 27.1750854
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.96
Output dim: 2, lower bound: -27.1750854, upper bound: 27.1750854
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.96
Output dim: 2, lower bound: -27.1769950, upper bound: 27.1769931
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.96
Output dim: 2, lower bound: -27.1769955, upper bound: 27.1769917
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 15.96
Output dim: 2, lower bound: -27.1710265, upper bound: 27.1710310
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 15.96
Output dim: 2, lower bound: -27.1710265, upper bound: 27.1710310

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1687455, upper bound: 27.1687335
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1687455, upper bound: 27.1687335
time: 7.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1740302, upper bound: 27.1740217
time: 10.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1740302, upper bound: 27.1740217
time: 7.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1751069, upper bound: 27.1751243
time: 6.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1751043, upper bound: 27.1751285
time: 3.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1645527, upper bound: 27.1645716
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1645527, upper bound: 27.1645716
time: 4.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1735549, upper bound: 27.1735410
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1735498, upper bound: 27.1735493
time: 3.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1735493, upper bound: 27.1735549
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1735489, upper bound: 27.1735549
time: 2.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1750843, upper bound: 27.1750854
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1750854, upper bound: 27.1750843
time: 5.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1724805, upper bound: 27.1724805
time: 6.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1724805, upper bound: 27.1724805
time: 5.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1768329, upper bound: 27.1768321
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1768329, upper bound: 27.1768321
time: 2.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1769953, upper bound: 27.1769916
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1769954, upper bound: 27.1769911
time: 7.04 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 12.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1687455, upper bound: 27.1687335
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1687455, upper bound: 27.1687335
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1740302, upper bound: 27.1740217
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1740302, upper bound: 27.1740217
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1751069, upper bound: 27.1751243
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1751043, upper bound: 27.1751285
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1645527, upper bound: 27.1645716
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1645527, upper bound: 27.1645716
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1735549, upper bound: 27.1735410
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1735498, upper bound: 27.1735493
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1735493, upper bound: 27.1735549
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1735489, upper bound: 27.1735549
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1750843, upper bound: 27.1750854
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1750854, upper bound: 27.1750843
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1724805, upper bound: 27.1724805
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1724805, upper bound: 27.1724805
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1768329, upper bound: 27.1768321
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1768329, upper bound: 27.1768321
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1769953, upper bound: 27.1769916
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 2, lower bound: -27.1769954, upper bound: 27.1769911

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1737411, upper bound: 27.1737299
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1737411, upper bound: 27.1737299
time: 5.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1740302, upper bound: 27.1740213
time: 2.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1740259, upper bound: 27.1740217
time: 4.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1653349, upper bound: 27.1653402
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1653344, upper bound: 27.1653401
time: 3.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1747406, upper bound: 27.1747577
time: 16.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1747406, upper bound: 27.1747577
time: 6.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1735549, upper bound: 27.1735410
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1735536, upper bound: 27.1735410
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1735496, upper bound: 27.1735489
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1735498, upper bound: 27.1735493
time: 8.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1695149, upper bound: 27.1695261
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1695149, upper bound: 27.1695261
time: 10.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1735489, upper bound: 27.1735304
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1735262, upper bound: 27.1735549
time: 4.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1642815, upper bound: 27.1642878
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1642815, upper bound: 27.1642878
time: 7.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1654979, upper bound: 27.1654871
time: 10.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1654979, upper bound: 27.1654871
time: 9.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1768329, upper bound: 27.1768319
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1768329, upper bound: 27.1768322
time: 5.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1489452, upper bound: 27.1489459
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1489452, upper bound: 27.1489459
time: 4.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1489432, upper bound: 27.1489450
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1489432, upper bound: 27.1489450
time: 6.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1722620, upper bound: 27.1722523
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1722624, upper bound: 27.1722523
time: 4.76 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 11.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1737411, upper bound: 27.1737299
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1737411, upper bound: 27.1737299
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1740302, upper bound: 27.1740213
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1740259, upper bound: 27.1740217
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1653349, upper bound: 27.1653402
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1653344, upper bound: 27.1653401
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1747406, upper bound: 27.1747577
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1747406, upper bound: 27.1747577
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1735549, upper bound: 27.1735410
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1735536, upper bound: 27.1735410
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1735496, upper bound: 27.1735489
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1735498, upper bound: 27.1735493
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1695149, upper bound: 27.1695261
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1695149, upper bound: 27.1695261
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1735489, upper bound: 27.1735304
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1735262, upper bound: 27.1735549
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1642815, upper bound: 27.1642878
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1642815, upper bound: 27.1642878
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1654979, upper bound: 27.1654871
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1654979, upper bound: 27.1654871
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1768329, upper bound: 27.1768319
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1768329, upper bound: 27.1768322
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1489452, upper bound: 27.1489459
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1489452, upper bound: 27.1489459
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1489432, upper bound: 27.1489450
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1489432, upper bound: 27.1489450
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1722620, upper bound: 27.1722523
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 11.66
Output dim: 2, lower bound: -27.1722624, upper bound: 27.1722523

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1737411, upper bound: 27.1737299
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1737406, upper bound: 27.1737299
time: 4.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1737366, upper bound: 27.1737299
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1737411, upper bound: 27.1737248
time: 6.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1615080, upper bound: 27.1614804
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1615088, upper bound: 27.1614809
time: 4.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1740233, upper bound: 27.1740217
time: 3.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1740259, upper bound: 27.1740189
time: 5.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1735096, upper bound: 27.1735323
time: 6.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1735080, upper bound: 27.1735337
time: 3.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1709851, upper bound: 27.1710221
time: 13.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1709851, upper bound: 27.1710221
time: 4.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 7.79 + 592.58 = 600.37 seconds
