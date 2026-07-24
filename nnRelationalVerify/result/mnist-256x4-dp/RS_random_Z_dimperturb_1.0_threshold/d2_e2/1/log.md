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
Threshold: 4.241187818999999


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890)
1: (-1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385)
2: (-2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155)
3: (-2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479)
4: (-2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695)
5: (-2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987)
6: (-2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276)
7: (-2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247)
8: (-3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120)
9: (-2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.25 + 3.43 = 4.68 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -4.2840282, upper bound: 4.2840282

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2840090, upper bound: 4.2840279
time: 3.00 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2840279, upper bound: 4.2840090
time: 2.05 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 5.07 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 5.07
Output dim: 8, lower bound: -4.2840090, upper bound: 4.2840279
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 5.07
Output dim: 8, lower bound: -4.2840279, upper bound: 4.2840090

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2199912, upper bound: 4.2201101
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2199912, upper bound: 4.2201101
time: 1.17 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2837651, upper bound: 4.2837491
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2837720, upper bound: 4.2837442
time: 3.75 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 6.64 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 6.64
Output dim: 8, lower bound: -4.2199912, upper bound: 4.2201101
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 6.64
Output dim: 8, lower bound: -4.2199912, upper bound: 4.2201101
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.64
Output dim: 8, lower bound: -4.2837651, upper bound: 4.2837491
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.64
Output dim: 8, lower bound: -4.2837720, upper bound: 4.2837442

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2821866, upper bound: 4.2821721
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2821933, upper bound: 4.2821687
time: 1.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2821874, upper bound: 4.2821722
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2821932, upper bound: 4.2821684
time: 1.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.53 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.53
Output dim: 8, lower bound: -4.2821866, upper bound: 4.2821721
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.53
Output dim: 8, lower bound: -4.2821933, upper bound: 4.2821687
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.53
Output dim: 8, lower bound: -4.2821874, upper bound: 4.2821722
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.53
Output dim: 8, lower bound: -4.2821932, upper bound: 4.2821684

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1613072, upper bound: 4.1611920
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1613072, upper bound: 4.1611920
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2200444, upper bound: 4.2199731
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2200444, upper bound: 4.2199731
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2809433, upper bound: 4.2812091
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2812857, upper bound: 4.2809244
time: 2.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2821303, upper bound: 4.2820852
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2821029, upper bound: 4.2821114
time: 1.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.99 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.99
Output dim: 8, lower bound: -4.1613072, upper bound: 4.1611920
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.99
Output dim: 8, lower bound: -4.1613072, upper bound: 4.1611920
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.99
Output dim: 8, lower bound: -4.2200444, upper bound: 4.2199731
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.99
Output dim: 8, lower bound: -4.2200444, upper bound: 4.2199731
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -4.2809433, upper bound: 4.2812091
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -4.2812857, upper bound: 4.2809244
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -4.2821303, upper bound: 4.2820852
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -4.2821029, upper bound: 4.2821114

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793006, upper bound: 4.2797271
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794421, upper bound: 4.2795776
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2808632, upper bound: 4.2808065
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811746, upper bound: 4.2806132
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2808843, upper bound: 4.2808398
time: 3.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2808843, upper bound: 4.2808398
time: 2.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2810102, upper bound: 4.2810158
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2810130, upper bound: 4.2810154
time: 3.58 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.29 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 8, lower bound: -4.2793006, upper bound: 4.2797271
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 8, lower bound: -4.2794421, upper bound: 4.2795776
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 8, lower bound: -4.2808632, upper bound: 4.2808065
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 8, lower bound: -4.2811746, upper bound: 4.2806132
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 8, lower bound: -4.2808843, upper bound: 4.2808398
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 8, lower bound: -4.2808843, upper bound: 4.2808398
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 8, lower bound: -4.2810102, upper bound: 4.2810158
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.29
Output dim: 8, lower bound: -4.2810130, upper bound: 4.2810154

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792282, upper bound: 4.2796106
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792111, upper bound: 4.2796624
time: 1.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.0942444, upper bound: 4.0942365
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.0942444, upper bound: 4.0942365
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2808109, upper bound: 4.2807245
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2807931, upper bound: 4.2807413
time: 2.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811164, upper bound: 4.2805546
time: 2.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2810786, upper bound: 4.2805626
time: 5.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2805379, upper bound: 4.2807383
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2807864, upper bound: 4.2805047
time: 4.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796057, upper bound: 4.2798478
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2799678, upper bound: 4.2795342
time: 4.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793792, upper bound: 4.2793996
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793792, upper bound: 4.2793996
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793924, upper bound: 4.2793977
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793924, upper bound: 4.2793977
time: 2.01 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.85 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -4.2792282, upper bound: 4.2796106
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -4.2792111, upper bound: 4.2796624
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.85
Output dim: 8, lower bound: -4.0942444, upper bound: 4.0942365
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.85
Output dim: 8, lower bound: -4.0942444, upper bound: 4.0942365
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -4.2808109, upper bound: 4.2807245
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -4.2807931, upper bound: 4.2807413
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -4.2811164, upper bound: 4.2805546
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -4.2810786, upper bound: 4.2805626
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -4.2805379, upper bound: 4.2807383
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -4.2807864, upper bound: 4.2805047
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -4.2796057, upper bound: 4.2798478
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -4.2799678, upper bound: 4.2795342
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -4.2793792, upper bound: 4.2793996
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -4.2793792, upper bound: 4.2793996
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -4.2793924, upper bound: 4.2793977
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 8, lower bound: -4.2793924, upper bound: 4.2793977

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.0940810, upper bound: 4.0940154
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.0940810, upper bound: 4.0940154
time: 1.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1552314, upper bound: 4.1551385
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1552314, upper bound: 4.1551385
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2798361, upper bound: 4.2798685
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2798361, upper bound: 4.2798685
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2795149, upper bound: 4.2794311
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2795149, upper bound: 4.2794311
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2748053, upper bound: 4.2743738
time: 5.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2748053, upper bound: 4.2743738
time: 4.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2798096, upper bound: 4.2791411
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2798096, upper bound: 4.2791539
time: 2.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791209, upper bound: 4.2792841
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791209, upper bound: 4.2792774
time: 3.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.0480822, upper bound: 4.0480177
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.0480822, upper bound: 4.0480177
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2783042, upper bound: 4.2785154
time: 2.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2783042, upper bound: 4.2785105
time: 2.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2782185, upper bound: 4.2779305
time: 2.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2784051, upper bound: 4.2777553
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2777642, upper bound: 4.2779073
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2778848, upper bound: 4.2777354
time: 2.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2790772, upper bound: 4.2793321
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793142, upper bound: 4.2790937
time: 1.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2782747, upper bound: 4.2785767
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2785924, upper bound: 4.2782534
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2723098, upper bound: 4.2723558
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2723098, upper bound: 4.2723558
time: 1.55 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.26 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.0940810, upper bound: 4.0940154
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.0940810, upper bound: 4.0940154
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.1552314, upper bound: 4.1551385
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.1552314, upper bound: 4.1551385
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2798361, upper bound: 4.2798685
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2798361, upper bound: 4.2798685
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2795149, upper bound: 4.2794311
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2795149, upper bound: 4.2794311
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2748053, upper bound: 4.2743738
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2748053, upper bound: 4.2743738
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2798096, upper bound: 4.2791411
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2798096, upper bound: 4.2791539
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2791209, upper bound: 4.2792841
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2791209, upper bound: 4.2792774
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.0480822, upper bound: 4.0480177
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.0480822, upper bound: 4.0480177
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2783042, upper bound: 4.2785154
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2783042, upper bound: 4.2785105
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2782185, upper bound: 4.2779305
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2784051, upper bound: 4.2777553
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2777642, upper bound: 4.2779073
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2778848, upper bound: 4.2777354
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2790772, upper bound: 4.2793321
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2793142, upper bound: 4.2790937
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2782747, upper bound: 4.2785767
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2785924, upper bound: 4.2782534
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2723098, upper bound: 4.2723558
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -4.2723098, upper bound: 4.2723558

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2732041, upper bound: 4.2731609
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2732041, upper bound: 4.2731609
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2783257, upper bound: 4.2784266
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2784638, upper bound: 4.2783302
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2777610, upper bound: 4.2778275
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2779321, upper bound: 4.2776765
time: 1.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2777610, upper bound: 4.2778275
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2779321, upper bound: 4.2776765
time: 2.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2728881, upper bound: 4.2725565
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2729693, upper bound: 4.2724873
time: 2.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2728881, upper bound: 4.2725565
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2729693, upper bound: 4.2724873
time: 2.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.9908131, upper bound: 3.9907144
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.9908131, upper bound: 3.9907144
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2785125, upper bound: 4.2779440
time: 2.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2785242, upper bound: 4.2779440
time: 3.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2722120, upper bound: 4.2722646
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2722120, upper bound: 4.2722646
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2775097, upper bound: 4.2777875
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2776517, upper bound: 4.2776393
time: 1.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2766601, upper bound: 4.2769904
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2767815, upper bound: 4.2768509
time: 2.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2780261, upper bound: 4.2784439
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2782365, upper bound: 4.2781920
time: 1.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2777786, upper bound: 4.2777950
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2781055, upper bound: 4.2775374
time: 1.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.9705914, upper bound: 3.9704418
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.9705914, upper bound: 3.9704418
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2701733, upper bound: 4.2702763
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2701733, upper bound: 4.2702763
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2702542, upper bound: 4.2702016
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2702542, upper bound: 4.2702016
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2774882, upper bound: 4.2778578
time: 2.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2776041, upper bound: 4.2776714
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2723098, upper bound: 4.2722104
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2723098, upper bound: 4.2722104
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2715615
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2715615
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2782460, upper bound: 4.2781820
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2785261, upper bound: 4.2779504
time: 1.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2715615
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715577, upper bound: 4.2712461
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2715615
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715577, upper bound: 4.2712461
time: 1.53 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 5.95 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2732041, upper bound: 4.2731609
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2732041, upper bound: 4.2731609
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2783257, upper bound: 4.2784266
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2784638, upper bound: 4.2783302
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2777610, upper bound: 4.2778275
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2779321, upper bound: 4.2776765
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2777610, upper bound: 4.2778275
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2779321, upper bound: 4.2776765
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2728881, upper bound: 4.2725565
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2729693, upper bound: 4.2724873
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2728881, upper bound: 4.2725565
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2729693, upper bound: 4.2724873
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.95
Output dim: 8, lower bound: -3.9908131, upper bound: 3.9907144
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.95
Output dim: 8, lower bound: -3.9908131, upper bound: 3.9907144
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2785125, upper bound: 4.2779440
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2785242, upper bound: 4.2779440
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2722120, upper bound: 4.2722646
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2722120, upper bound: 4.2722646
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2775097, upper bound: 4.2777875
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2776517, upper bound: 4.2776393
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2766601, upper bound: 4.2769904
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2767815, upper bound: 4.2768509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2780261, upper bound: 4.2784439
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2782365, upper bound: 4.2781920
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2777786, upper bound: 4.2777950
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2781055, upper bound: 4.2775374
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.95
Output dim: 8, lower bound: -3.9705914, upper bound: 3.9704418
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.95
Output dim: 8, lower bound: -3.9705914, upper bound: 3.9704418
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2701733, upper bound: 4.2702763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2701733, upper bound: 4.2702763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2702542, upper bound: 4.2702016
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2702542, upper bound: 4.2702016
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2774882, upper bound: 4.2778578
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2776041, upper bound: 4.2776714
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2723098, upper bound: 4.2722104
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2723098, upper bound: 4.2722104
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2715615
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2715615
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2782460, upper bound: 4.2781820
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2785261, upper bound: 4.2779504
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2715615
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2715577, upper bound: 4.2712461
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2715615
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.95
Output dim: 8, lower bound: -4.2715577, upper bound: 4.2712461

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2711524, upper bound: 4.2712009
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2712474, upper bound: 4.2711461
time: 2.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2714746, upper bound: 4.2711680
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2714746, upper bound: 4.2711680
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2711524, upper bound: 4.2712009
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2711524, upper bound: 4.2712009
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1367823, upper bound: 4.1367182
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1367823, upper bound: 4.1367182
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2766095, upper bound: 4.2766257
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2766095, upper bound: 4.2766257
time: 1.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2767359, upper bound: 4.2765027
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2767359, upper bound: 4.2765027
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717158, upper bound: 4.2717016
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717158, upper bound: 4.2717016
time: 1.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2767359, upper bound: 4.2765027
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2767359, upper bound: 4.2765027
time: 1.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2712976, upper bound: 4.2710676
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2712976, upper bound: 4.2710676
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2713890, upper bound: 4.2710085
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2713890, upper bound: 4.2710085
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2719355, upper bound: 4.2715261
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2719355, upper bound: 4.2715261
time: 2.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2713890, upper bound: 4.2710085
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2713890, upper bound: 4.2710085
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2768446, upper bound: 4.2763843
time: 2.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2770001, upper bound: 4.2762827
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715568, upper bound: 4.2710905
time: 2.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715568, upper bound: 4.2710905
time: 2.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2711602, upper bound: 4.2714387
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2714469, upper bound: 4.2711588
time: 1.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2711602, upper bound: 4.2714387
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2714469, upper bound: 4.2711588
time: 1.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2700801, upper bound: 4.2701911
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2700801, upper bound: 4.2701911
time: 2.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2701461, upper bound: 4.2701113
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2701461, upper bound: 4.2701113
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2763642, upper bound: 4.2768990
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2765719, upper bound: 4.2766394
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2691841, upper bound: 4.2692347
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2691841, upper bound: 4.2692347
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2711602, upper bound: 4.2714387
time: 2.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2711602, upper bound: 4.2714387
time: 2.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2713540, upper bound: 4.2712547
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2713540, upper bound: 4.2712547
time: 1.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.9697713, upper bound: 3.9696930
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.9697713, upper bound: 3.9696930
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2719356, upper bound: 4.2715281
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2719356, upper bound: 4.2715281
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2700346, upper bound: 4.2702763
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2701733, upper bound: 4.2701159
time: 2.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2700346, upper bound: 4.2702763
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2701733, upper bound: 4.2701159
time: 2.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2691133, upper bound: 4.2693351
time: 2.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2694156, upper bound: 4.2690543
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2691133, upper bound: 4.2693351
time: 3.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2694156, upper bound: 4.2690543
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2763471, upper bound: 4.2770049
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2766106, upper bound: 4.2766254
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2764338, upper bound: 4.2768086
time: 2.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2767326, upper bound: 4.2764765
time: 2.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2701733, upper bound: 4.2701159
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2702542, upper bound: 4.2700437
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2713763
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715577, upper bound: 4.2710961
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2711267, upper bound: 4.2715615
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2713763
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2711267, upper bound: 4.2715615
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2713763
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2713996, upper bound: 4.2712461
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2713996, upper bound: 4.2712461
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2768700, upper bound: 4.2764012
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2770073, upper bound: 4.2762824
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2711267, upper bound: 4.2715615
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2713763
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2693707, upper bound: 4.2690942
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2694156, upper bound: 4.2690543
time: 1.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2711267, upper bound: 4.2715615
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2713763
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2693707, upper bound: 4.2690942
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2694156, upper bound: 4.2690543
time: 1.51 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 6.67 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2711524, upper bound: 4.2712009
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2712474, upper bound: 4.2711461
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2714746, upper bound: 4.2711680
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2714746, upper bound: 4.2711680
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2711524, upper bound: 4.2712009
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2711524, upper bound: 4.2712009
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.1367823, upper bound: 4.1367182
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.1367823, upper bound: 4.1367182
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2766095, upper bound: 4.2766257
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2766095, upper bound: 4.2766257
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2767359, upper bound: 4.2765027
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2767359, upper bound: 4.2765027
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2717158, upper bound: 4.2717016
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2717158, upper bound: 4.2717016
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2767359, upper bound: 4.2765027
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2767359, upper bound: 4.2765027
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2712976, upper bound: 4.2710676
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2712976, upper bound: 4.2710676
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2713890, upper bound: 4.2710085
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2713890, upper bound: 4.2710085
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2719355, upper bound: 4.2715261
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2719355, upper bound: 4.2715261
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2713890, upper bound: 4.2710085
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2713890, upper bound: 4.2710085
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2768446, upper bound: 4.2763843
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2770001, upper bound: 4.2762827
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2715568, upper bound: 4.2710905
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2715568, upper bound: 4.2710905
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2711602, upper bound: 4.2714387
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2714469, upper bound: 4.2711588
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2711602, upper bound: 4.2714387
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2714469, upper bound: 4.2711588
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2700801, upper bound: 4.2701911
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2700801, upper bound: 4.2701911
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2701461, upper bound: 4.2701113
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2701461, upper bound: 4.2701113
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2763642, upper bound: 4.2768990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2765719, upper bound: 4.2766394
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2691841, upper bound: 4.2692347
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2691841, upper bound: 4.2692347
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2711602, upper bound: 4.2714387
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2711602, upper bound: 4.2714387
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2713540, upper bound: 4.2712547
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2713540, upper bound: 4.2712547
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.67
Output dim: 8, lower bound: -3.9697713, upper bound: 3.9696930
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.67
Output dim: 8, lower bound: -3.9697713, upper bound: 3.9696930
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2719356, upper bound: 4.2715281
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2719356, upper bound: 4.2715281
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2700346, upper bound: 4.2702763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2701733, upper bound: 4.2701159
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2700346, upper bound: 4.2702763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2701733, upper bound: 4.2701159
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2691133, upper bound: 4.2693351
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2694156, upper bound: 4.2690543
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2691133, upper bound: 4.2693351
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2694156, upper bound: 4.2690543
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2763471, upper bound: 4.2770049
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2766106, upper bound: 4.2766254
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2764338, upper bound: 4.2768086
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2767326, upper bound: 4.2764765
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2701733, upper bound: 4.2701159
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2702542, upper bound: 4.2700437
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2713763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2715577, upper bound: 4.2710961
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2711267, upper bound: 4.2715615
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2713763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2711267, upper bound: 4.2715615
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2713763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2713996, upper bound: 4.2712461
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2713996, upper bound: 4.2712461
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2768700, upper bound: 4.2764012
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2770073, upper bound: 4.2762824
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2711267, upper bound: 4.2715615
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2713763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2693707, upper bound: 4.2690942
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2694156, upper bound: 4.2690543
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2711267, upper bound: 4.2715615
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2713763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2693707, upper bound: 4.2690942
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.67
Output dim: 8, lower bound: -4.2694156, upper bound: 4.2690543

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2692723, upper bound: 4.2690186
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2692723, upper bound: 4.2690186
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2693259, upper bound: 4.2689945
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2693259, upper bound: 4.2689945
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2692723, upper bound: 4.2690186
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2693259, upper bound: 4.2689945
time: 1.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2692723, upper bound: 4.2690186
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2693259, upper bound: 4.2689945
time: 1.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2692723, upper bound: 4.2690186
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2692723, upper bound: 4.2690186
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890
1: -1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385
2: -2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155
3: -2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479
4: -2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695
5: -2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987
6: -2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276
7: -2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247
8: -3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120
9: -2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2692723, upper bound: 4.2690186
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2692723, upper bound: 4.2690186
time: 1.39 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 6.11 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 6.11
Output dim: 8, lower bound: -4.2692723, upper bound: 4.2690186
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 6.11
Output dim: 8, lower bound: -4.2692723, upper bound: 4.2690186
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 6.11
Output dim: 8, lower bound: -4.2693259, upper bound: 4.2689945
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 6.11
Output dim: 8, lower bound: -4.2693259, upper bound: 4.2689945
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 6.11
Output dim: 8, lower bound: -4.2692723, upper bound: 4.2690186
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 6.11
Output dim: 8, lower bound: -4.2693259, upper bound: 4.2689945
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 6.11
Output dim: 8, lower bound: -4.2692723, upper bound: 4.2690186
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 6.11
Output dim: 8, lower bound: -4.2693259, upper bound: 4.2689945
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 6.11
Output dim: 8, lower bound: -4.2692723, upper bound: 4.2690186
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 6.11
Output dim: 8, lower bound: -4.2692723, upper bound: 4.2690186
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 6.11
Output dim: 8, lower bound: -4.2692723, upper bound: 4.2690186
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 6.11
Output dim: 8, lower bound: -4.2692723, upper bound: 4.2690186
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2766095, upper bound: 4.2766257
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2766095, upper bound: 4.2766257
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2767359, upper bound: 4.2765027
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2767359, upper bound: 4.2765027
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2717158, upper bound: 4.2717016
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2717158, upper bound: 4.2717016
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2767359, upper bound: 4.2765027
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2767359, upper bound: 4.2765027
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2712976, upper bound: 4.2710676
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2712976, upper bound: 4.2710676
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2713890, upper bound: 4.2710085
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2713890, upper bound: 4.2710085
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2719355, upper bound: 4.2715261
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2719355, upper bound: 4.2715261
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2713890, upper bound: 4.2710085
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2713890, upper bound: 4.2710085
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2768446, upper bound: 4.2763843
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2770001, upper bound: 4.2762827
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2715568, upper bound: 4.2710905
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2715568, upper bound: 4.2710905
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2711602, upper bound: 4.2714387
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2714469, upper bound: 4.2711588
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2711602, upper bound: 4.2714387
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2714469, upper bound: 4.2711588
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2700801, upper bound: 4.2701911
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2700801, upper bound: 4.2701911
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2701461, upper bound: 4.2701113
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2701461, upper bound: 4.2701113
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2763642, upper bound: 4.2768990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2765719, upper bound: 4.2766394
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2691841, upper bound: 4.2692347
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2691841, upper bound: 4.2692347
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2711602, upper bound: 4.2714387
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2711602, upper bound: 4.2714387
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2713540, upper bound: 4.2712547
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2713540, upper bound: 4.2712547
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2719356, upper bound: 4.2715281
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2719356, upper bound: 4.2715281
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2700346, upper bound: 4.2702763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2701733, upper bound: 4.2701159
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2700346, upper bound: 4.2702763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2701733, upper bound: 4.2701159
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2691133, upper bound: 4.2693351
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2694156, upper bound: 4.2690543
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2691133, upper bound: 4.2693351
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2694156, upper bound: 4.2690543
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2763471, upper bound: 4.2770049
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2766106, upper bound: 4.2766254
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2764338, upper bound: 4.2768086
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2767326, upper bound: 4.2764765
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2701733, upper bound: 4.2701159
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2702542, upper bound: 4.2700437
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2713763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2715577, upper bound: 4.2710961
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2711267, upper bound: 4.2715615
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2713763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2711267, upper bound: 4.2715615
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2713763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2713996, upper bound: 4.2712461
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2713996, upper bound: 4.2712461
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2768700, upper bound: 4.2764012
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2770073, upper bound: 4.2762824
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2711267, upper bound: 4.2715615
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2713763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2693707, upper bound: 4.2690942
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2694156, upper bound: 4.2690543
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2711267, upper bound: 4.2715615
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2712920, upper bound: 4.2713763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2693707, upper bound: 4.2690942
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.11
Output dim: 8, lower bound: -4.2694156, upper bound: 4.2690543

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 4.68 + 596.19 = 600.87 seconds
