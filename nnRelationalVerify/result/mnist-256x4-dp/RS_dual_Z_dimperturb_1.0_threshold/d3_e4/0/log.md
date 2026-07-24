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
execution time: IAR + RelationalAnalysis = 1.42 + 6.41 = 7.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -27.2005054, upper bound: 27.2005054

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987716, upper bound: 27.1987630
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987630, upper bound: 27.1987716
time: 16.44 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 21.87 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 21.87
Output dim: 2, lower bound: -27.1987716, upper bound: 27.1987630
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 21.87
Output dim: 2, lower bound: -27.1987630, upper bound: 27.1987716

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976968, upper bound: 27.1976668
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976679, upper bound: 27.1976835
time: 11.92 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976834, upper bound: 27.1976679
time: 10.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976668, upper bound: 27.1976968
time: 3.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.73 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.73
Output dim: 2, lower bound: -27.1976968, upper bound: 27.1976668
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.73
Output dim: 2, lower bound: -27.1976679, upper bound: 27.1976835
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.73
Output dim: 2, lower bound: -27.1976834, upper bound: 27.1976679
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.73
Output dim: 2, lower bound: -27.1976668, upper bound: 27.1976968

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976968, upper bound: 27.1976668
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976968, upper bound: 27.1976667
time: 4.03 seconds

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976679, upper bound: 27.1976834
time: 7.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976679, upper bound: 27.1976834
time: 5.43 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976834, upper bound: 27.1976679
time: 6.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976834, upper bound: 27.1976679
time: 7.72 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976667, upper bound: 27.1976968
time: 18.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976668, upper bound: 27.1976968
time: 5.79 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.29 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.29
Output dim: 2, lower bound: -27.1976968, upper bound: 27.1976668
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.29
Output dim: 2, lower bound: -27.1976968, upper bound: 27.1976667
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.29
Output dim: 2, lower bound: -27.1976679, upper bound: 27.1976834
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.29
Output dim: 2, lower bound: -27.1976679, upper bound: 27.1976834
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.29
Output dim: 2, lower bound: -27.1976834, upper bound: 27.1976679
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.29
Output dim: 2, lower bound: -27.1976834, upper bound: 27.1976679
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.29
Output dim: 2, lower bound: -27.1976667, upper bound: 27.1976968
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.29
Output dim: 2, lower bound: -27.1976668, upper bound: 27.1976968

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918442, upper bound: 27.1918391
time: 11.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918442, upper bound: 27.1918391
time: 8.84 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918450, upper bound: 27.1918380
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918450, upper bound: 27.1918380
time: 14.80 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918388, upper bound: 27.1918436
time: 10.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918388, upper bound: 27.1918436
time: 6.00 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918398, upper bound: 27.1918427
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918398, upper bound: 27.1918427
time: 2.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918427, upper bound: 27.1918398
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918427, upper bound: 27.1918398
time: 4.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918436, upper bound: 27.1918388
time: 7.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918436, upper bound: 27.1918388
time: 4.09 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918380, upper bound: 27.1918450
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918380, upper bound: 27.1918450
time: 5.18 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918391, upper bound: 27.1918442
time: 5.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918391, upper bound: 27.1918442
time: 6.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 13.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 2, lower bound: -27.1918442, upper bound: 27.1918391
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 2, lower bound: -27.1918442, upper bound: 27.1918391
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 2, lower bound: -27.1918450, upper bound: 27.1918380
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 2, lower bound: -27.1918450, upper bound: 27.1918380
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 2, lower bound: -27.1918388, upper bound: 27.1918436
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 2, lower bound: -27.1918388, upper bound: 27.1918436
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 2, lower bound: -27.1918398, upper bound: 27.1918427
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 2, lower bound: -27.1918398, upper bound: 27.1918427
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 2, lower bound: -27.1918427, upper bound: 27.1918398
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 2, lower bound: -27.1918427, upper bound: 27.1918398
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 2, lower bound: -27.1918436, upper bound: 27.1918388
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 2, lower bound: -27.1918436, upper bound: 27.1918388
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 2, lower bound: -27.1918380, upper bound: 27.1918450
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 2, lower bound: -27.1918380, upper bound: 27.1918450
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 2, lower bound: -27.1918391, upper bound: 27.1918442
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 2, lower bound: -27.1918391, upper bound: 27.1918442

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754811
time: 2.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754811
time: 2.80 seconds

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
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754811
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754811
time: 6.92 seconds

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
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754806
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754806
time: 5.37 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754806
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754806
time: 4.31 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
time: 11.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
time: 12.38 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
time: 4.50 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754851
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754851
time: 5.14 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754852
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754852
time: 5.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
time: 8.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
time: 7.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
time: 9.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
time: 5.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
time: 4.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
time: 3.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
time: 3.05 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
time: 9.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
time: 11.71 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
time: 14.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
time: 12.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
time: 15.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
time: 3.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
time: 4.79 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 9.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754811
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754811
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754811
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754811
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754806
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754806
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754806
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754806
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754851
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754851
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754852
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754852
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.42
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754807
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754851, upper bound: 27.1754811
time: 13.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754807
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754851, upper bound: 27.1754811
time: 10.47 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754807
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754851, upper bound: 27.1754811
time: 5.05 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754807
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754851, upper bound: 27.1754811
time: 5.03 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754801
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754806
time: 3.12 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754801
time: 2.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754806
time: 3.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754801
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754806
time: 13.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754801
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754806
time: 6.62 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754852
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754799, upper bound: 27.1754854
time: 2.60 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 8.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.25
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754807
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.25
Output dim: 2, lower bound: -27.1754851, upper bound: 27.1754811
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.25
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754807
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.25
Output dim: 2, lower bound: -27.1754851, upper bound: 27.1754811
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.25
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754807
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.25
Output dim: 2, lower bound: -27.1754851, upper bound: 27.1754811
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.25
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754807
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.25
Output dim: 2, lower bound: -27.1754851, upper bound: 27.1754811
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.25
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754801
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.25
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754806
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.25
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754801
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.25
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754806
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.25
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754801
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.25
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754806
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.25
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754801
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.25
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754806
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.25
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754852
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.25
Output dim: 2, lower bound: -27.1754799, upper bound: 27.1754854
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754851
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754851
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754852
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754852
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.25
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 7.83 + 597.87 = 605.70 seconds
