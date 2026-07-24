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
execution time: IAR + RelationalAnalysis = 1.29 + 3.41 = 4.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -4.2840282, upper bound: 4.2840282

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2827872, upper bound: 4.2827872
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2827872, upper bound: 4.2827872
time: 5.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.07 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.07
Output dim: 8, lower bound: -4.2827872, upper bound: 4.2827872
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.07
Output dim: 8, lower bound: -4.2827872, upper bound: 4.2827872

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2810962, upper bound: 4.2811020
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2810962
time: 1.42 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2810962, upper bound: 4.2811020
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2810962
time: 2.72 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.43 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.43
Output dim: 8, lower bound: -4.2810962, upper bound: 4.2811020
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.43
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2810962
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.43
Output dim: 8, lower bound: -4.2810962, upper bound: 4.2811020
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.43
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2810962

## BFS RS instance: RS_RSZ1_RSZ1

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2795244, upper bound: 4.2797038
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2795244
time: 2.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2795244, upper bound: 4.2797038
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2795244
time: 2.21 seconds

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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2795244, upper bound: 4.2797038
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2795244
time: 2.19 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2795244, upper bound: 4.2797038
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2795244
time: 2.20 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.17 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.17
Output dim: 8, lower bound: -4.2795244, upper bound: 4.2797038
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.17
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2795244
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.17
Output dim: 8, lower bound: -4.2795244, upper bound: 4.2797038
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.17
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2795244
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.17
Output dim: 8, lower bound: -4.2795244, upper bound: 4.2797038
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.17
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2795244
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.17
Output dim: 8, lower bound: -4.2795244, upper bound: 4.2797038
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.17
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2795244

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794757, upper bound: 4.2796055
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794548, upper bound: 4.2796534
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796534, upper bound: 4.2794572
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796055, upper bound: 4.2794757
time: 2.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794757, upper bound: 4.2796055
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794572, upper bound: 4.2796534
time: 2.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796534, upper bound: 4.2794548
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796055, upper bound: 4.2794757
time: 1.50 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794757, upper bound: 4.2796055
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794548, upper bound: 4.2796534
time: 1.33 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796534, upper bound: 4.2794572
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796055, upper bound: 4.2794757
time: 2.33 seconds

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
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794757, upper bound: 4.2796055
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794572, upper bound: 4.2796534
time: 1.37 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796534, upper bound: 4.2794548
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796055, upper bound: 4.2794757
time: 1.49 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -4.2794757, upper bound: 4.2796055
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -4.2794548, upper bound: 4.2796534
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -4.2796534, upper bound: 4.2794572
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -4.2796055, upper bound: 4.2794757
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -4.2794757, upper bound: 4.2796055
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -4.2794572, upper bound: 4.2796534
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -4.2796534, upper bound: 4.2794548
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -4.2796055, upper bound: 4.2794757
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -4.2794757, upper bound: 4.2796055
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -4.2794548, upper bound: 4.2796534
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -4.2796534, upper bound: 4.2794572
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -4.2796055, upper bound: 4.2794757
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -4.2794757, upper bound: 4.2796055
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -4.2794572, upper bound: 4.2796534
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -4.2796534, upper bound: 4.2794548
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -4.2796055, upper bound: 4.2794757

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791518, upper bound: 4.2792647
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791540, upper bound: 4.2792547
time: 2.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791300, upper bound: 4.2793102
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791330, upper bound: 4.2793068
time: 2.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793068, upper bound: 4.2791340
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793102, upper bound: 4.2791307
time: 2.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792547, upper bound: 4.2791548
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792647, upper bound: 4.2791533
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791533, upper bound: 4.2792647
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791548, upper bound: 4.2792547
time: 1.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791307, upper bound: 4.2793102
time: 3.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791340, upper bound: 4.2793068
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793068, upper bound: 4.2791330
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793102, upper bound: 4.2791300
time: 2.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792547, upper bound: 4.2791540
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792647, upper bound: 4.2791518
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791518, upper bound: 4.2792647
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791540, upper bound: 4.2792547
time: 2.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791300, upper bound: 4.2793102
time: 2.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791330, upper bound: 4.2793068
time: 4.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793068, upper bound: 4.2791340
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793102, upper bound: 4.2791307
time: 2.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792547, upper bound: 4.2791548
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792647, upper bound: 4.2791533
time: 1.43 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791533, upper bound: 4.2792647
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791548, upper bound: 4.2792547
time: 1.66 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791307, upper bound: 4.2793102
time: 3.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791340, upper bound: 4.2793068
time: 1.76 seconds

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793068, upper bound: 4.2791330
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793102, upper bound: 4.2791300
time: 2.22 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792547, upper bound: 4.2791540
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792647, upper bound: 4.2791518
time: 1.41 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.17 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2791518, upper bound: 4.2792647
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2791540, upper bound: 4.2792547
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2791300, upper bound: 4.2793102
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2791330, upper bound: 4.2793068
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2793068, upper bound: 4.2791340
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2793102, upper bound: 4.2791307
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2792547, upper bound: 4.2791548
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2792647, upper bound: 4.2791533
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2791533, upper bound: 4.2792647
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2791548, upper bound: 4.2792547
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2791307, upper bound: 4.2793102
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2791340, upper bound: 4.2793068
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2793068, upper bound: 4.2791330
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2793102, upper bound: 4.2791300
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2792547, upper bound: 4.2791540
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2792647, upper bound: 4.2791518
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2791518, upper bound: 4.2792647
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2791540, upper bound: 4.2792547
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2791300, upper bound: 4.2793102
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2791330, upper bound: 4.2793068
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2793068, upper bound: 4.2791340
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2793102, upper bound: 4.2791307
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2792547, upper bound: 4.2791548
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2792647, upper bound: 4.2791533
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2791533, upper bound: 4.2792647
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2791548, upper bound: 4.2792547
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2791307, upper bound: 4.2793102
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2791340, upper bound: 4.2793068
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2793068, upper bound: 4.2791330
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2793102, upper bound: 4.2791300
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2792547, upper bound: 4.2791540
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 8, lower bound: -4.2792647, upper bound: 4.2791518

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
time: 2.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
time: 1.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
time: 3.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
time: 2.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
time: 1.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
time: 2.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
time: 2.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
time: 2.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
time: 1.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
time: 2.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
time: 2.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
time: 2.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
time: 1.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
time: 2.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
time: 2.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
time: 1.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
time: 1.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
time: 3.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
time: 4.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
time: 2.75 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
time: 2.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
time: 2.44 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
time: 1.78 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
time: 1.94 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
time: 1.46 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
time: 2.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
time: 2.77 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
time: 3.87 seconds

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
time: 1.54 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
time: 1.49 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 6.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
time: 4.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
time: 4.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716306, upper bound: 4.2717547
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716306, upper bound: 4.2717547
time: 1.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717547, upper bound: 4.2716306
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718088, upper bound: 4.2715940
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717547, upper bound: 4.2716306
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718088, upper bound: 4.2715940
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717546, upper bound: 4.2716294
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2715921
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717546, upper bound: 4.2716294
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2715921
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716698, upper bound: 4.2716983
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716635
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716698, upper bound: 4.2716983
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716635
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716726, upper bound: 4.2716994
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717340, upper bound: 4.2716635
time: 2.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716726, upper bound: 4.2716994
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717340, upper bound: 4.2716635
time: 2.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
time: 3.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
time: 1.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
time: 1.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
time: 2.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
time: 2.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
time: 3.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716306, upper bound: 4.2717547
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716306, upper bound: 4.2717547
time: 1.41 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 6.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716306, upper bound: 4.2717547
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716306, upper bound: 4.2717547
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2717547, upper bound: 4.2716306
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2718088, upper bound: 4.2715940
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2717547, upper bound: 4.2716306
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2718088, upper bound: 4.2715940
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2717546, upper bound: 4.2716294
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2715921
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2717546, upper bound: 4.2716294
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2715921
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716698, upper bound: 4.2716983
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716635
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716698, upper bound: 4.2716983
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716635
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716726, upper bound: 4.2716994
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2717340, upper bound: 4.2716635
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716726, upper bound: 4.2716994
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2717340, upper bound: 4.2716635
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716306, upper bound: 4.2717547
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 8, lower bound: -4.2716306, upper bound: 4.2717547
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.05
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 4.70 + 597.58 = 602.28 seconds
