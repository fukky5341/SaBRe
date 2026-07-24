## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 6.43e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006704, 0.0006704)
1: (-0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001262, 0.0001262)
2: (0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007597, 0.0007597)
3: (1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003)
4: (-0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0001006, 0.0001006)
5: (0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0005049, 0.0005049)
6: (-0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440)
7: (-0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017233, 0.0017233)
8: (-0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0009400, 0.0009400)
9: (0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003885, 0.0003885)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.46 + 1.41 = 2.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0000912, upper bound: 0.0000913

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000867, upper bound: 0.0000776
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000776, upper bound: 0.0000868
time: 0.53 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.36 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.36
Output dim: 3, lower bound: -0.0000867, upper bound: 0.0000776
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.36
Output dim: 3, lower bound: -0.0000776, upper bound: 0.0000868

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006702, 0.0006703
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001261, 0.0001261
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007594, 0.0007595
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0001006, 0.0001006
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0005047, 0.0005047
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017233, 0.0017233
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0009394, 0.0009392
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003880, 0.0003881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000833, upper bound: 0.0000688
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000779, upper bound: 0.0000737
time: 0.61 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006703, 0.0006702
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001261, 0.0001261
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007595, 0.0007594
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0001006, 0.0001006
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0005047, 0.0005047
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017233, 0.0017233
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0009392, 0.0009394
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003881, 0.0003880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000737, upper bound: 0.0000779
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000687, upper bound: 0.0000833
time: 0.60 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 3, lower bound: -0.0000833, upper bound: 0.0000688
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 3, lower bound: -0.0000779, upper bound: 0.0000737
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 3, lower bound: -0.0000737, upper bound: 0.0000779
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 3, lower bound: -0.0000687, upper bound: 0.0000833

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006684, 0.0006688
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001261, 0.0001263
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007564, 0.0007571
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0001001, 0.0001000
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0005033, 0.0005036
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017232, 0.0017231
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0009328, 0.0009313
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003834, 0.0003842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000770, upper bound: 0.0000640
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000782, upper bound: 0.0000635
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006688, 0.0006684
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001263, 0.0001261
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007570, 0.0007565
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0001000, 0.0001001
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0005036, 0.0005033
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017231, 0.0017232
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0009315, 0.0009326
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003841, 0.0003835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000710, upper bound: 0.0000685
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000728, upper bound: 0.0000677
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006684, 0.0006688
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001261, 0.0001263
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007565, 0.0007570
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0001001, 0.0001000
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0005033, 0.0005036
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017232, 0.0017231
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0009326, 0.0009315
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003835, 0.0003841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000678, upper bound: 0.0000728
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000685, upper bound: 0.0000710
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006688, 0.0006684
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001263, 0.0001261
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007571, 0.0007564
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0001000, 0.0001001
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0005036, 0.0005033
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017231, 0.0017232
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0009313, 0.0009328
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003842, 0.0003834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000635, upper bound: 0.0000782
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000639, upper bound: 0.0000771
time: 0.61 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.32 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 3, lower bound: -0.0000770, upper bound: 0.0000640
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 3, lower bound: -0.0000782, upper bound: 0.0000635
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 3, lower bound: -0.0000710, upper bound: 0.0000685
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 3, lower bound: -0.0000728, upper bound: 0.0000677
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 3, lower bound: -0.0000678, upper bound: 0.0000728
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 3, lower bound: -0.0000685, upper bound: 0.0000710
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 3, lower bound: -0.0000635, upper bound: 0.0000782
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 3, lower bound: -0.0000639, upper bound: 0.0000771

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006746, 0.0006605
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001299, 0.0001230
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007659, 0.0007443
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000977, 0.0001017
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0005081, 0.0004971
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017222, 0.0017238
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0009071, 0.0009539
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003971, 0.0003720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000758, upper bound: 0.0000604
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000739, upper bound: 0.0000627
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006601, 0.0006688
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001228, 0.0001263
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007436, 0.0007571
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0001001, 0.0000976
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004967, 0.0005036
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017232, 0.0017221
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0009328, 0.0009056
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003713, 0.0003842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000769, upper bound: 0.0000600
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000749, upper bound: 0.0000622
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006749, 0.0006601
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001300, 0.0001228
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007663, 0.0007437
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000976, 0.0001018
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0005083, 0.0004968
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017221, 0.0017239
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0009058, 0.0009548
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003975, 0.0003713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000636
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000696, upper bound: 0.0000671
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006605, 0.0006684
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001230, 0.0001261
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007442, 0.0007565
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0001000, 0.0000977
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004970, 0.0005033
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017231, 0.0017222
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0009315, 0.0009069
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003720, 0.0003835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000717, upper bound: 0.0000633
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000712, upper bound: 0.0000663
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006747, 0.0006605
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001299, 0.0001230
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007660, 0.0007442
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000977, 0.0001017
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0005082, 0.0004970
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017222, 0.0017239
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0009069, 0.0009542
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003972, 0.0003720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000663, upper bound: 0.0000712
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000633, upper bound: 0.0000717
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006601, 0.0006688
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001228, 0.0001263
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007437, 0.0007570
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0001001, 0.0000976
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004968, 0.0005036
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017232, 0.0017221
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0009326, 0.0009058
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003713, 0.0003841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000671, upper bound: 0.0000696
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000636, upper bound: 0.0000699
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006749, 0.0006601
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001301, 0.0001228
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007664, 0.0007436
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000976, 0.0001018
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0005084, 0.0004967
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017221, 0.0017239
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0009056, 0.0009550
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003977, 0.0003713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000622, upper bound: 0.0000749
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000600, upper bound: 0.0000769
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006605, 0.0006684
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001230, 0.0001261
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007443, 0.0007564
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0001000, 0.0000977
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004971, 0.0005033
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017231, 0.0017222
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0009313, 0.0009071
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003720, 0.0003834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000627, upper bound: 0.0000739
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000604, upper bound: 0.0000758
time: 0.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 3, lower bound: -0.0000758, upper bound: 0.0000604
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 3, lower bound: -0.0000739, upper bound: 0.0000627
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 3, lower bound: -0.0000769, upper bound: 0.0000600
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 3, lower bound: -0.0000749, upper bound: 0.0000622
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000636
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 3, lower bound: -0.0000696, upper bound: 0.0000671
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 3, lower bound: -0.0000717, upper bound: 0.0000633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 3, lower bound: -0.0000712, upper bound: 0.0000663
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 3, lower bound: -0.0000663, upper bound: 0.0000712
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 3, lower bound: -0.0000633, upper bound: 0.0000717
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 3, lower bound: -0.0000671, upper bound: 0.0000696
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 3, lower bound: -0.0000636, upper bound: 0.0000699
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 3, lower bound: -0.0000622, upper bound: 0.0000749
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 3, lower bound: -0.0000600, upper bound: 0.0000769
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 3, lower bound: -0.0000627, upper bound: 0.0000739
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 3, lower bound: -0.0000604, upper bound: 0.0000758

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006577, 0.0006462
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001269, 0.0001213
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007407, 0.0007231
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000938, 0.0000971
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004948, 0.0004858
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017204, 0.0017218
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008593, 0.0008976
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003621, 0.0003416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000658, upper bound: 0.0000545
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000545
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006599, 0.0006436
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001280, 0.0001200
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007442, 0.0007191
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000931, 0.0000977
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004966, 0.0004838
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017201, 0.0017220
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008507, 0.0009050
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003660, 0.0003370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000553
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000556
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006432, 0.0006544
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001198, 0.0001242
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007185, 0.0007352
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000960, 0.0000929
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004834, 0.0004922
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017214, 0.0017201
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008829, 0.0008493
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003363, 0.0003532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000658, upper bound: 0.0000545
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000545
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006458, 0.0006518
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001211, 0.0001229
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007225, 0.0007312
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000952, 0.0000937
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004855, 0.0004902
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017211, 0.0017204
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008743, 0.0008580
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003409, 0.0003487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000553
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000556
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006579, 0.0006458
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001270, 0.0001211
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007411, 0.0007225
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000937, 0.0000972
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004950, 0.0004855
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017204, 0.0017218
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008581, 0.0008985
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003625, 0.0003410

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000632, upper bound: 0.0000566
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000625, upper bound: 0.0000567
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006603, 0.0006432
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001282, 0.0001198
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007448, 0.0007185
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000929, 0.0000978
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004969, 0.0004835
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017201, 0.0017221
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008494, 0.0009063
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003667, 0.0003363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000632, upper bound: 0.0000573
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000626, upper bound: 0.0000583
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006436, 0.0006540
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001200, 0.0001240
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007190, 0.0007346
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000959, 0.0000930
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004837, 0.0004920
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017214, 0.0017201
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008817, 0.0008506
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003370, 0.0003526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000632, upper bound: 0.0000566
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000625, upper bound: 0.0000567
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006461, 0.0006514
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001213, 0.0001227
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007230, 0.0007306
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000951, 0.0000938
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004858, 0.0004899
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017211, 0.0017204
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008730, 0.0008592
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003416, 0.0003480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000632, upper bound: 0.0000573
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000626, upper bound: 0.0000582
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006577, 0.0006461
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001269, 0.0001213
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007409, 0.0007230
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000938, 0.0000971
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004949, 0.0004858
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017204, 0.0017218
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008592, 0.0008979
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003622, 0.0003416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000583, upper bound: 0.0000626
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000573, upper bound: 0.0000632
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006599, 0.0006436
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001280, 0.0001200
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007442, 0.0007190
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000930, 0.0000977
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004966, 0.0004837
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017201, 0.0017220
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008506, 0.0009052
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003661, 0.0003370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000567, upper bound: 0.0000625
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000566, upper bound: 0.0000632
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006432, 0.0006544
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001198, 0.0001242
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007185, 0.0007351
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000960, 0.0000929
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004835, 0.0004922
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017214, 0.0017201
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008828, 0.0008494
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003363, 0.0003532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000583, upper bound: 0.0000626
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000573, upper bound: 0.0000632
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006458, 0.0006518
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001211, 0.0001229
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007225, 0.0007312
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000952, 0.0000937
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004855, 0.0004902
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017211, 0.0017204
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008742, 0.0008581
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003410, 0.0003486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000567, upper bound: 0.0000625
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000566, upper bound: 0.0000632
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006580, 0.0006458
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001271, 0.0001211
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007412, 0.0007225
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000937, 0.0000972
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004951, 0.0004855
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017204, 0.0017218
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008580, 0.0008987
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003627, 0.0003409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000556, upper bound: 0.0000643
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000656
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006603, 0.0006432
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001282, 0.0001198
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007448, 0.0007185
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000929, 0.0000979
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004969, 0.0004834
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017201, 0.0017221
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008493, 0.0009065
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003668, 0.0003363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000545, upper bound: 0.0000643
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000545, upper bound: 0.0000658
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006436, 0.0006540
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001200, 0.0001240
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007191, 0.0007346
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000959, 0.0000931
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004838, 0.0004919
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017214, 0.0017201
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008816, 0.0008507
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003370, 0.0003525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000556, upper bound: 0.0000643
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000655
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006462, 0.0006514
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001213, 0.0001227
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007231, 0.0007306
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000951, 0.0000938
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004858, 0.0004899
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017211, 0.0017204
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008729, 0.0008593
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003416, 0.0003479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000545, upper bound: 0.0000643
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000545, upper bound: 0.0000658
time: 0.56 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000658, upper bound: 0.0000545
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000545
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000553
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000556
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000658, upper bound: 0.0000545
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000545
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000553
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000556
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000632, upper bound: 0.0000566
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000625, upper bound: 0.0000567
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000632, upper bound: 0.0000573
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000626, upper bound: 0.0000583
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000632, upper bound: 0.0000566
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000625, upper bound: 0.0000567
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000632, upper bound: 0.0000573
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000626, upper bound: 0.0000582
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000583, upper bound: 0.0000626
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000573, upper bound: 0.0000632
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000567, upper bound: 0.0000625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000566, upper bound: 0.0000632
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000583, upper bound: 0.0000626
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000573, upper bound: 0.0000632
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000567, upper bound: 0.0000625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000566, upper bound: 0.0000632
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000556, upper bound: 0.0000643
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000656
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000545, upper bound: 0.0000643
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000545, upper bound: 0.0000658
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000556, upper bound: 0.0000643
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000655
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000545, upper bound: 0.0000643
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -0.0000545, upper bound: 0.0000658

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006539, 0.0006438
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001263, 0.0001213
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007351, 0.0007196
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000932, 0.0000961
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004919, 0.0004839
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017201, 0.0017213
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008528, 0.0008864
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003566, 0.0003386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000543, upper bound: 0.0000431
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000543, upper bound: 0.0000435
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006561, 0.0006410
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001273, 0.0001200
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007385, 0.0007154
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000924, 0.0000967
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004936, 0.0004818
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017198, 0.0017216
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008436, 0.0008938
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003606, 0.0003338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000439
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000440
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006394, 0.0006520
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001192, 0.0001242
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007128, 0.0007317
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000954, 0.0000920
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004805, 0.0004904
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017211, 0.0017196
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008764, 0.0008381
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003308, 0.0003503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000543, upper bound: 0.0000431
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000543, upper bound: 0.0000435
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006420, 0.0006493
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001204, 0.0001229
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007168, 0.0007275
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000946, 0.0000927
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004825, 0.0004882
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017208, 0.0017199
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008672, 0.0008468
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003354, 0.0003454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000439
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000440
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006554, 0.0006420
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001270, 0.0001204
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007375, 0.0007168
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000927, 0.0000966
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004931, 0.0004825
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017199, 0.0017215
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008468, 0.0008915
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003594, 0.0003354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000440, upper bound: 0.0000538
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000439, upper bound: 0.0000538
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006579, 0.0006394
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001282, 0.0001192
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007412, 0.0007128
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000920, 0.0000973
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004950, 0.0004805
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017196, 0.0017218
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008381, 0.0008997
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003637, 0.0003308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000435, upper bound: 0.0000543
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000431, upper bound: 0.0000543
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006410, 0.0006502
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001200, 0.0001234
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007154, 0.0007289
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000949, 0.0000924
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004818, 0.0004890
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017209, 0.0017198
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008704, 0.0008436
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003338, 0.0003471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000440, upper bound: 0.0000538
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000439, upper bound: 0.0000538
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000310, 0.0009241, -0.0000310, 0.0009241, -0.0006438, 0.0006476
1: -0.0034883, -0.0033131, -0.0034883, -0.0033131, -0.0001213, 0.0001221
2: 0.0148971, 0.0161004, 0.0148971, 0.0161004, -0.0007196, 0.0007249
3: 1.0066766, 1.0069768, 1.0066766, 1.0069768, -0.0003003, 0.0003003
4: -0.0042658, -0.0040816, -0.0042658, -0.0040816, -0.0000941, 0.0000932
5: 0.0039562, 0.0046855, 0.0039562, 0.0046855, -0.0004839, 0.0004869
6: -0.0026066, -0.0025626, -0.0026066, -0.0025626, -0.0000440, 0.0000440
7: -0.0131117, -0.0113574, -0.0131117, -0.0113574, -0.0017206, 0.0017201
8: -0.0138689, -0.0119540, -0.0138689, -0.0119540, -0.0008617, 0.0008528
9: 0.0018090, 0.0027111, 0.0018090, 0.0027111, -0.0003386, 0.0003424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000435, upper bound: 0.0000543
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000431, upper bound: 0.0000543
time: 0.63 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000543, upper bound: 0.0000431
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000543, upper bound: 0.0000435
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000439
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000440
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000543, upper bound: 0.0000431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000543, upper bound: 0.0000435
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000439
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000440
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000440, upper bound: 0.0000538
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000439, upper bound: 0.0000538
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000435, upper bound: 0.0000543
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000431, upper bound: 0.0000543
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000440, upper bound: 0.0000538
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000439, upper bound: 0.0000538
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000435, upper bound: 0.0000543
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000431, upper bound: 0.0000543

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.87 + 104.21 = 107.08 seconds
