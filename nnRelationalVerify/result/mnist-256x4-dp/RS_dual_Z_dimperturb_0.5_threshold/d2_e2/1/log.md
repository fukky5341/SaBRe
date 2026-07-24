## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.35483528


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.2127902, 0.2240139, -0.2127902, 0.2240139, -0.4340229, 0.4340229)
1: (-0.1270260, 0.1068145, -0.1270260, 0.1068145, -0.2338404, 0.2338404)
2: (-0.1302294, 0.1440212, -0.1302294, 0.1440212, -0.2742506, 0.2742506)
3: (-0.1040969, 0.1337786, -0.1040969, 0.1337786, -0.2378754, 0.2378754)
4: (-0.1261479, 0.0868855, -0.1261479, 0.0868855, -0.2130334, 0.2130334)
5: (-0.1445761, 0.1338020, -0.1445761, 0.1338020, -0.2783780, 0.2783780)
6: (-0.1735323, 0.1249607, -0.1735323, 0.1249607, -0.2946548, 0.2946547)
7: (-0.1261005, 0.1110680, -0.1261005, 0.1110680, -0.2371685, 0.2371685)
8: (0.4513179, 1.1703138, 0.4513179, 1.1703138, -0.6958580, 0.6958578)
9: (-0.1002106, 0.1508961, -0.1002106, 0.1508961, -0.2511067, 0.2511067)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.14 + 1.87 = 3.00 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.4248989, upper bound: 0.4248989

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4192883, upper bound: 0.4227277
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4227277, upper bound: 0.4192883
time: 0.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.69 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 8, lower bound: -0.4192883, upper bound: 0.4227277
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 8, lower bound: -0.4227277, upper bound: 0.4192883

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.2127902, 0.2240139, -0.2127902, 0.2240139, -0.4335506, 0.4333749
1: -0.1270260, 0.1068145, -0.1270260, 0.1068145, -0.2338404, 0.2338404
2: -0.1302294, 0.1440212, -0.1302294, 0.1440212, -0.2742506, 0.2742506
3: -0.1040969, 0.1337786, -0.1040969, 0.1337786, -0.2378754, 0.2378754
4: -0.1261479, 0.0868855, -0.1261479, 0.0868855, -0.2130334, 0.2130334
5: -0.1445761, 0.1338020, -0.1445761, 0.1338020, -0.2783780, 0.2783780
6: -0.1735323, 0.1249607, -0.1735323, 0.1249607, -0.2942978, 0.2941893
7: -0.1261005, 0.1110680, -0.1261005, 0.1110680, -0.2371685, 0.2371685
8: 0.4513179, 1.1703138, 0.4513179, 1.1703138, -0.6900225, 0.6913581
9: -0.1002106, 0.1508961, -0.1002106, 0.1508961, -0.2511067, 0.2511067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4041258, upper bound: 0.4177852
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4142026, upper bound: 0.4062318
time: 0.70 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.2127902, 0.2240139, -0.2127902, 0.2240139, -0.4333749, 0.4335506
1: -0.1270260, 0.1068145, -0.1270260, 0.1068145, -0.2338404, 0.2338404
2: -0.1302294, 0.1440212, -0.1302294, 0.1440212, -0.2742506, 0.2742506
3: -0.1040969, 0.1337786, -0.1040969, 0.1337786, -0.2378754, 0.2378754
4: -0.1261479, 0.0868855, -0.1261479, 0.0868855, -0.2130334, 0.2130334
5: -0.1445761, 0.1338020, -0.1445761, 0.1338020, -0.2783780, 0.2783780
6: -0.1735323, 0.1249607, -0.1735323, 0.1249607, -0.2941893, 0.2942977
7: -0.1261005, 0.1110680, -0.1261005, 0.1110680, -0.2371685, 0.2371685
8: 0.4513179, 1.1703138, 0.4513179, 1.1703138, -0.6913581, 0.6900225
9: -0.1002106, 0.1508961, -0.1002106, 0.1508961, -0.2511067, 0.2511067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4062318, upper bound: 0.4142026
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4177852, upper bound: 0.4041258
time: 0.71 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.74 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.4041258, upper bound: 0.4177852
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.4142026, upper bound: 0.4062318
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.4062318, upper bound: 0.4142026
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.4177852, upper bound: 0.4041258

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2127902, 0.2240139, -0.2127902, 0.2240139, -0.4335446, 0.4333673
1: -0.1270260, 0.1068145, -0.1270260, 0.1068145, -0.2338404, 0.2338404
2: -0.1302294, 0.1440212, -0.1302294, 0.1440212, -0.2742506, 0.2742506
3: -0.1040969, 0.1337786, -0.1040969, 0.1337786, -0.2378754, 0.2378754
4: -0.1261479, 0.0868855, -0.1261479, 0.0868855, -0.2130334, 0.2130334
5: -0.1445761, 0.1338020, -0.1445761, 0.1338020, -0.2783780, 0.2783780
6: -0.1735323, 0.1249607, -0.1735323, 0.1249607, -0.2942941, 0.2941847
7: -0.1261005, 0.1110680, -0.1261005, 0.1110680, -0.2371685, 0.2371685
8: 0.4513179, 1.1703138, 0.4513179, 1.1703138, -0.6899662, 0.6913135
9: -0.1002106, 0.1508961, -0.1002106, 0.1508961, -0.2511067, 0.2511067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3912398, upper bound: 0.4017747
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3912398, upper bound: 0.4017747
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2127902, 0.2240139, -0.2127902, 0.2240139, -0.4335432, 0.4333687
1: -0.1270260, 0.1068145, -0.1270260, 0.1068145, -0.2338404, 0.2338404
2: -0.1302294, 0.1440212, -0.1302294, 0.1440212, -0.2742506, 0.2742506
3: -0.1040969, 0.1337786, -0.1040969, 0.1337786, -0.2378754, 0.2378754
4: -0.1261479, 0.0868855, -0.1261479, 0.0868855, -0.2130334, 0.2130334
5: -0.1445761, 0.1338020, -0.1445761, 0.1338020, -0.2783780, 0.2783780
6: -0.1735323, 0.1249607, -0.1735323, 0.1249607, -0.2942932, 0.2941855
7: -0.1261005, 0.1110680, -0.1261005, 0.1110680, -0.2371685, 0.2371685
8: 0.4513179, 1.1703138, 0.4513179, 1.1703138, -0.6899767, 0.6913021
9: -0.1002106, 0.1508961, -0.1002106, 0.1508961, -0.2511067, 0.2511067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3997349, upper bound: 0.3928385
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3997349, upper bound: 0.3928385
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2127902, 0.2240139, -0.2127902, 0.2240139, -0.4333689, 0.4335432
1: -0.1270260, 0.1068145, -0.1270260, 0.1068145, -0.2338404, 0.2338404
2: -0.1302294, 0.1440212, -0.1302294, 0.1440212, -0.2742506, 0.2742506
3: -0.1040969, 0.1337786, -0.1040969, 0.1337786, -0.2378754, 0.2378754
4: -0.1261479, 0.0868855, -0.1261479, 0.0868855, -0.2130334, 0.2130334
5: -0.1445761, 0.1338020, -0.1445761, 0.1338020, -0.2783780, 0.2783780
6: -0.1735323, 0.1249607, -0.1735323, 0.1249607, -0.2941856, 0.2942930
7: -0.1261005, 0.1110680, -0.1261005, 0.1110680, -0.2371685, 0.2371685
8: 0.4513179, 1.1703138, 0.4513179, 1.1703138, -0.6913018, 0.6899769
9: -0.1002106, 0.1508961, -0.1002106, 0.1508961, -0.2511067, 0.2511067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3928385, upper bound: 0.3997349
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3928385, upper bound: 0.3997349
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2127902, 0.2240139, -0.2127902, 0.2240139, -0.4333675, 0.4335446
1: -0.1270260, 0.1068145, -0.1270260, 0.1068145, -0.2338404, 0.2338404
2: -0.1302294, 0.1440212, -0.1302294, 0.1440212, -0.2742506, 0.2742506
3: -0.1040969, 0.1337786, -0.1040969, 0.1337786, -0.2378754, 0.2378754
4: -0.1261479, 0.0868855, -0.1261479, 0.0868855, -0.2130334, 0.2130334
5: -0.1445761, 0.1338020, -0.1445761, 0.1338020, -0.2783780, 0.2783780
6: -0.1735323, 0.1249607, -0.1735323, 0.1249607, -0.2941847, 0.2942940
7: -0.1261005, 0.1110680, -0.1261005, 0.1110680, -0.2371685, 0.2371685
8: 0.4513179, 1.1703138, 0.4513179, 1.1703138, -0.6913137, 0.6899660
9: -0.1002106, 0.1508961, -0.1002106, 0.1508961, -0.2511067, 0.2511067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4017747, upper bound: 0.3912398
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4017747, upper bound: 0.3912398
time: 0.86 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.96 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 8, lower bound: -0.3912398, upper bound: 0.4017747
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 8, lower bound: -0.3912398, upper bound: 0.4017747
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 8, lower bound: -0.3997349, upper bound: 0.3928385
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 8, lower bound: -0.3997349, upper bound: 0.3928385
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 8, lower bound: -0.3928385, upper bound: 0.3997349
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 8, lower bound: -0.3928385, upper bound: 0.3997349
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 8, lower bound: -0.4017747, upper bound: 0.3912398
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 8, lower bound: -0.4017747, upper bound: 0.3912398

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2127902, 0.2240139, -0.2127902, 0.2240139, -0.4334300, 0.4331408
1: -0.1270260, 0.1068145, -0.1270260, 0.1068145, -0.2338404, 0.2338404
2: -0.1302294, 0.1440212, -0.1302294, 0.1440212, -0.2742506, 0.2742506
3: -0.1040969, 0.1337786, -0.1040969, 0.1337786, -0.2378754, 0.2378754
4: -0.1261479, 0.0868855, -0.1261479, 0.0868855, -0.2130334, 0.2130334
5: -0.1445761, 0.1338020, -0.1445761, 0.1338020, -0.2783780, 0.2783780
6: -0.1735323, 0.1249607, -0.1735323, 0.1249607, -0.2942220, 0.2940434
7: -0.1261005, 0.1110680, -0.1261005, 0.1110680, -0.2371685, 0.2371685
8: 0.4513179, 1.1703138, 0.4513179, 1.1703138, -0.6881890, 0.6903889
9: -0.1002106, 0.1508961, -0.1002106, 0.1508961, -0.2511067, 0.2511067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3405515, upper bound: 0.3461474
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3405515, upper bound: 0.3461474
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2127902, 0.2240139, -0.2127902, 0.2240139, -0.4335446, 0.4332528
1: -0.1270260, 0.1068145, -0.1270260, 0.1068145, -0.2338404, 0.2338404
2: -0.1302294, 0.1440212, -0.1302294, 0.1440212, -0.2742506, 0.2742506
3: -0.1040969, 0.1337786, -0.1040969, 0.1337786, -0.2378754, 0.2378754
4: -0.1261479, 0.0868855, -0.1261479, 0.0868855, -0.2130334, 0.2130334
5: -0.1445761, 0.1338020, -0.1445761, 0.1338020, -0.2783780, 0.2783780
6: -0.1735323, 0.1249607, -0.1735323, 0.1249607, -0.2942941, 0.2941126
7: -0.1261005, 0.1110680, -0.1261005, 0.1110680, -0.2371685, 0.2371685
8: 0.4513179, 1.1703138, 0.4513179, 1.1703138, -0.6890411, 0.6913135
9: -0.1002106, 0.1508961, -0.1002106, 0.1508961, -0.2511067, 0.2511067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3405515, upper bound: 0.3461474
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3405515, upper bound: 0.3461474
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2127902, 0.2240139, -0.2127902, 0.2240139, -0.4334285, 0.4331417
1: -0.1270260, 0.1068145, -0.1270260, 0.1068145, -0.2338404, 0.2338404
2: -0.1302294, 0.1440212, -0.1302294, 0.1440212, -0.2742506, 0.2742506
3: -0.1040969, 0.1337786, -0.1040969, 0.1337786, -0.2378754, 0.2378754
4: -0.1261479, 0.0868855, -0.1261479, 0.0868855, -0.2130334, 0.2130334
5: -0.1445761, 0.1338020, -0.1445761, 0.1338020, -0.2783780, 0.2783780
6: -0.1735323, 0.1249607, -0.1735323, 0.1249607, -0.2942210, 0.2940443
7: -0.1261005, 0.1110680, -0.1261005, 0.1110680, -0.2371685, 0.2371685
8: 0.4513179, 1.1703138, 0.4513179, 1.1703138, -0.6881986, 0.6903770
9: -0.1002106, 0.1508961, -0.1002106, 0.1508961, -0.2511067, 0.2511067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3458896, upper bound: 0.3408329
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3458896, upper bound: 0.3408329
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2127902, 0.2240139, -0.2127902, 0.2240139, -0.4335432, 0.4332542
1: -0.1270260, 0.1068145, -0.1270260, 0.1068145, -0.2338404, 0.2338404
2: -0.1302294, 0.1440212, -0.1302294, 0.1440212, -0.2742506, 0.2742506
3: -0.1040969, 0.1337786, -0.1040969, 0.1337786, -0.2378754, 0.2378754
4: -0.1261479, 0.0868855, -0.1261479, 0.0868855, -0.2130334, 0.2130334
5: -0.1445761, 0.1338020, -0.1445761, 0.1338020, -0.2783780, 0.2783780
6: -0.1735323, 0.1249607, -0.1735323, 0.1249607, -0.2942932, 0.2941134
7: -0.1261005, 0.1110680, -0.1261005, 0.1110680, -0.2371685, 0.2371685
8: 0.4513179, 1.1703138, 0.4513179, 1.1703138, -0.6890516, 0.6913021
9: -0.1002106, 0.1508961, -0.1002106, 0.1508961, -0.2511067, 0.2511067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3458896, upper bound: 0.3408329
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3458896, upper bound: 0.3408329
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2127902, 0.2240139, -0.2127902, 0.2240139, -0.4332540, 0.4332976
1: -0.1270260, 0.1068145, -0.1270260, 0.1068145, -0.2338404, 0.2338404
2: -0.1302294, 0.1440212, -0.1302294, 0.1440212, -0.2742506, 0.2742506
3: -0.1040969, 0.1337786, -0.1040969, 0.1337786, -0.2378754, 0.2378754
4: -0.1261479, 0.0868855, -0.1261479, 0.0868855, -0.2130334, 0.2130334
5: -0.1445761, 0.1338020, -0.1445761, 0.1338020, -0.2783780, 0.2783780
6: -0.1735323, 0.1249607, -0.1735323, 0.1249607, -0.2941135, 0.2941402
7: -0.1261005, 0.1110680, -0.1261005, 0.1110680, -0.2371685, 0.2371685
8: 0.4513179, 1.1703138, 0.4513179, 1.1703138, -0.6893826, 0.6890519
9: -0.1002106, 0.1508961, -0.1002106, 0.1508961, -0.2511067, 0.2511067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3408329, upper bound: 0.3458896
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3408329, upper bound: 0.3458896
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2127902, 0.2240139, -0.2127902, 0.2240139, -0.4333689, 0.4334283
1: -0.1270260, 0.1068145, -0.1270260, 0.1068145, -0.2338404, 0.2338404
2: -0.1302294, 0.1440212, -0.1302294, 0.1440212, -0.2742506, 0.2742506
3: -0.1040969, 0.1337786, -0.1040969, 0.1337786, -0.2378754, 0.2378754
4: -0.1261479, 0.0868855, -0.1261479, 0.0868855, -0.2130334, 0.2130334
5: -0.1445761, 0.1338020, -0.1445761, 0.1338020, -0.2783780, 0.2783780
6: -0.1735323, 0.1249607, -0.1735323, 0.1249607, -0.2941856, 0.2942209
7: -0.1261005, 0.1110680, -0.1261005, 0.1110680, -0.2371685, 0.2371685
8: 0.4513179, 1.1703138, 0.4513179, 1.1703138, -0.6903772, 0.6899769
9: -0.1002106, 0.1508961, -0.1002106, 0.1508961, -0.2511067, 0.2511067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3408329, upper bound: 0.3458896
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3408329, upper bound: 0.3458896
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2127902, 0.2240139, -0.2127902, 0.2240139, -0.4332528, 0.4332991
1: -0.1270260, 0.1068145, -0.1270260, 0.1068145, -0.2338404, 0.2338404
2: -0.1302294, 0.1440212, -0.1302294, 0.1440212, -0.2742506, 0.2742506
3: -0.1040969, 0.1337786, -0.1040969, 0.1337786, -0.2378754, 0.2378754
4: -0.1261479, 0.0868855, -0.1261479, 0.0868855, -0.2130334, 0.2130334
5: -0.1445761, 0.1338020, -0.1445761, 0.1338020, -0.2783780, 0.2783780
6: -0.1735323, 0.1249607, -0.1735323, 0.1249607, -0.2941126, 0.2941411
7: -0.1261005, 0.1110680, -0.1261005, 0.1110680, -0.2371685, 0.2371685
8: 0.4513179, 1.1703138, 0.4513179, 1.1703138, -0.6893930, 0.6890414
9: -0.1002106, 0.1508961, -0.1002106, 0.1508961, -0.2511067, 0.2511067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3461474, upper bound: 0.3405515
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3461474, upper bound: 0.3405515
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2127902, 0.2240139, -0.2127902, 0.2240139, -0.4333675, 0.4334300
1: -0.1270260, 0.1068145, -0.1270260, 0.1068145, -0.2338404, 0.2338404
2: -0.1302294, 0.1440212, -0.1302294, 0.1440212, -0.2742506, 0.2742506
3: -0.1040969, 0.1337786, -0.1040969, 0.1337786, -0.2378754, 0.2378754
4: -0.1261479, 0.0868855, -0.1261479, 0.0868855, -0.2130334, 0.2130334
5: -0.1445761, 0.1338020, -0.1445761, 0.1338020, -0.2783780, 0.2783780
6: -0.1735323, 0.1249607, -0.1735323, 0.1249607, -0.2941847, 0.2942219
7: -0.1261005, 0.1110680, -0.1261005, 0.1110680, -0.2371685, 0.2371685
8: 0.4513179, 1.1703138, 0.4513179, 1.1703138, -0.6903887, 0.6899660
9: -0.1002106, 0.1508961, -0.1002106, 0.1508961, -0.2511067, 0.2511067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3461474, upper bound: 0.3405515
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3461474, upper bound: 0.3405515
time: 0.62 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.35
Output dim: 8, lower bound: -0.3405515, upper bound: 0.3461474
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.35
Output dim: 8, lower bound: -0.3405515, upper bound: 0.3461474
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.35
Output dim: 8, lower bound: -0.3405515, upper bound: 0.3461474
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.35
Output dim: 8, lower bound: -0.3405515, upper bound: 0.3461474
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.35
Output dim: 8, lower bound: -0.3458896, upper bound: 0.3408329
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.35
Output dim: 8, lower bound: -0.3458896, upper bound: 0.3408329
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.35
Output dim: 8, lower bound: -0.3458896, upper bound: 0.3408329
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.35
Output dim: 8, lower bound: -0.3458896, upper bound: 0.3408329
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.35
Output dim: 8, lower bound: -0.3408329, upper bound: 0.3458896
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.35
Output dim: 8, lower bound: -0.3408329, upper bound: 0.3458896
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.35
Output dim: 8, lower bound: -0.3408329, upper bound: 0.3458896
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.35
Output dim: 8, lower bound: -0.3408329, upper bound: 0.3458896
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.35
Output dim: 8, lower bound: -0.3461474, upper bound: 0.3405515
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.35
Output dim: 8, lower bound: -0.3461474, upper bound: 0.3405515
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.35
Output dim: 8, lower bound: -0.3461474, upper bound: 0.3405515
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.35
Output dim: 8, lower bound: -0.3461474, upper bound: 0.3405515

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.00 + 55.39 = 58.39 seconds
