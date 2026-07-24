## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 4.241187818999999


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.18 + 3.43 = 5.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -4.2840282, upper bound: 4.2840282

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2827872, upper bound: 4.2827872
time: 4.54 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2827872, upper bound: 4.2827872
time: 5.24 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.98 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.98
Output dim: 8, lower bound: -4.2827872, upper bound: 4.2827872
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.98
Output dim: 8, lower bound: -4.2827872, upper bound: 4.2827872

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2810962, upper bound: 4.2811020
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2810962
time: 1.42 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2810962, upper bound: 4.2811020
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2810962
time: 2.94 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 6.46 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.46
Output dim: 8, lower bound: -4.2810962, upper bound: 4.2811020
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.46
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2810962
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.46
Output dim: 8, lower bound: -4.2810962, upper bound: 4.2811020
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.46
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2810962

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2795244, upper bound: 4.2797038
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2795244
time: 2.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2795244, upper bound: 4.2797038
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2795244
time: 2.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2795244, upper bound: 4.2797038
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2795244
time: 2.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2795244, upper bound: 4.2797038
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2795244
time: 2.20 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.51 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.51
Output dim: 8, lower bound: -4.2795244, upper bound: 4.2797038
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.51
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2795244
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.51
Output dim: 8, lower bound: -4.2795244, upper bound: 4.2797038
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.51
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2795244
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.51
Output dim: 8, lower bound: -4.2795244, upper bound: 4.2797038
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.51
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2795244
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.51
Output dim: 8, lower bound: -4.2795244, upper bound: 4.2797038
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.51
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2795244

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794757, upper bound: 4.2796055
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794548, upper bound: 4.2796534
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796534, upper bound: 4.2794572
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796055, upper bound: 4.2794757
time: 2.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794757, upper bound: 4.2796055
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794572, upper bound: 4.2796534
time: 2.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796534, upper bound: 4.2794548
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796055, upper bound: 4.2794757
time: 1.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794757, upper bound: 4.2796055
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794548, upper bound: 4.2796534
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796534, upper bound: 4.2794572
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796055, upper bound: 4.2794757
time: 2.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794757, upper bound: 4.2796055
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794572, upper bound: 4.2796534
time: 1.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796534, upper bound: 4.2794548
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796055, upper bound: 4.2794757
time: 1.52 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.00 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 8, lower bound: -4.2794757, upper bound: 4.2796055
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 8, lower bound: -4.2794548, upper bound: 4.2796534
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 8, lower bound: -4.2796534, upper bound: 4.2794572
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 8, lower bound: -4.2796055, upper bound: 4.2794757
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 8, lower bound: -4.2794757, upper bound: 4.2796055
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 8, lower bound: -4.2794572, upper bound: 4.2796534
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 8, lower bound: -4.2796534, upper bound: 4.2794548
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 8, lower bound: -4.2796055, upper bound: 4.2794757
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 8, lower bound: -4.2794757, upper bound: 4.2796055
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 8, lower bound: -4.2794548, upper bound: 4.2796534
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 8, lower bound: -4.2796534, upper bound: 4.2794572
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 8, lower bound: -4.2796055, upper bound: 4.2794757
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 8, lower bound: -4.2794757, upper bound: 4.2796055
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 8, lower bound: -4.2794572, upper bound: 4.2796534
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 8, lower bound: -4.2796534, upper bound: 4.2794548
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 8, lower bound: -4.2796055, upper bound: 4.2794757

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791518, upper bound: 4.2792647
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791540, upper bound: 4.2792547
time: 2.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791300, upper bound: 4.2793102
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791330, upper bound: 4.2793068
time: 2.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793068, upper bound: 4.2791340
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793102, upper bound: 4.2791307
time: 2.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792547, upper bound: 4.2791548
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792647, upper bound: 4.2791533
time: 1.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791533, upper bound: 4.2792647
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791548, upper bound: 4.2792547
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791307, upper bound: 4.2793102
time: 3.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791340, upper bound: 4.2793068
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793068, upper bound: 4.2791330
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793102, upper bound: 4.2791300
time: 2.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792547, upper bound: 4.2791540
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792647, upper bound: 4.2791518
time: 1.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791518, upper bound: 4.2792647
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791540, upper bound: 4.2792547
time: 2.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791300, upper bound: 4.2793102
time: 2.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791330, upper bound: 4.2793068
time: 4.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793068, upper bound: 4.2791340
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793102, upper bound: 4.2791307
time: 2.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792547, upper bound: 4.2791548
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792647, upper bound: 4.2791533
time: 1.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791533, upper bound: 4.2792647
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791548, upper bound: 4.2792547
time: 1.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791307, upper bound: 4.2793102
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791340, upper bound: 4.2793068
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793068, upper bound: 4.2791330
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793102, upper bound: 4.2791300
time: 2.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792547, upper bound: 4.2791540
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792647, upper bound: 4.2791518
time: 1.58 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 7.07 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2791518, upper bound: 4.2792647
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2791540, upper bound: 4.2792547
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2791300, upper bound: 4.2793102
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2791330, upper bound: 4.2793068
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2793068, upper bound: 4.2791340
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2793102, upper bound: 4.2791307
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2792547, upper bound: 4.2791548
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2792647, upper bound: 4.2791533
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2791533, upper bound: 4.2792647
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2791548, upper bound: 4.2792547
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2791307, upper bound: 4.2793102
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2791340, upper bound: 4.2793068
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2793068, upper bound: 4.2791330
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2793102, upper bound: 4.2791300
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2792547, upper bound: 4.2791540
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2792647, upper bound: 4.2791518
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2791518, upper bound: 4.2792647
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2791540, upper bound: 4.2792547
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2791300, upper bound: 4.2793102
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2791330, upper bound: 4.2793068
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2793068, upper bound: 4.2791340
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2793102, upper bound: 4.2791307
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2792547, upper bound: 4.2791548
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2792647, upper bound: 4.2791533
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2791533, upper bound: 4.2792647
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2791548, upper bound: 4.2792547
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2791307, upper bound: 4.2793102
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2791340, upper bound: 4.2793068
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2793068, upper bound: 4.2791330
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2793102, upper bound: 4.2791300
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2792547, upper bound: 4.2791540
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.07
Output dim: 8, lower bound: -4.2792647, upper bound: 4.2791518

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
time: 2.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
time: 3.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
time: 1.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
time: 4.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
time: 2.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
time: 1.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
time: 2.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
time: 2.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
time: 2.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
time: 1.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
time: 2.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
time: 2.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
time: 4.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
time: 2.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
time: 1.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
time: 1.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
time: 2.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
time: 2.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
time: 1.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
time: 1.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
time: 3.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
time: 3.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
time: 1.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
time: 2.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
time: 2.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
time: 1.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
time: 1.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
time: 2.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
time: 3.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
time: 3.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
time: 1.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
time: 1.49 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.58 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
time: 4.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
time: 4.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
time: 1.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
time: 1.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716306, upper bound: 4.2717547
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716306, upper bound: 4.2717547
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717547, upper bound: 4.2716306
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718088, upper bound: 4.2715940
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717547, upper bound: 4.2716306
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718088, upper bound: 4.2715940
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717546, upper bound: 4.2716294
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2715921
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717546, upper bound: 4.2716294
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2715921
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716698, upper bound: 4.2716983
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716635
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716698, upper bound: 4.2716983
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716635
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716726, upper bound: 4.2716994
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717340, upper bound: 4.2716635
time: 2.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716726, upper bound: 4.2716994
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717340, upper bound: 4.2716635
time: 2.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
time: 1.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
time: 1.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
time: 2.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
time: 3.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
time: 2.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
time: 1.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716306, upper bound: 4.2717547
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716306, upper bound: 4.2717547
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717547, upper bound: 4.2716306
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718088, upper bound: 4.2715940
time: 1.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717547, upper bound: 4.2716306
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718088, upper bound: 4.2715940
time: 1.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717546, upper bound: 4.2716294
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2715921
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717546, upper bound: 4.2716294
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2715921
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716698, upper bound: 4.2716983
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716635
time: 1.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716698, upper bound: 4.2716983
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716635
time: 1.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716726, upper bound: 4.2716994
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717340, upper bound: 4.2716635
time: 1.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716726, upper bound: 4.2716994
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717340, upper bound: 4.2716635
time: 1.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
time: 1.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
time: 1.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
time: 1.78 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 5.13 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716306, upper bound: 4.2717547
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716306, upper bound: 4.2717547
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2717547, upper bound: 4.2716306
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2718088, upper bound: 4.2715940
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2717547, upper bound: 4.2716306
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2718088, upper bound: 4.2715940
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2717546, upper bound: 4.2716294
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2715921
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2717546, upper bound: 4.2716294
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2715921
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716698, upper bound: 4.2716983
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716635
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716698, upper bound: 4.2716983
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716635
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716726, upper bound: 4.2716994
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2717340, upper bound: 4.2716635
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716726, upper bound: 4.2716994
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2717340, upper bound: 4.2716635
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716306, upper bound: 4.2717547
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716306, upper bound: 4.2717547
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2717547, upper bound: 4.2716306
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2718088, upper bound: 4.2715940
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2717547, upper bound: 4.2716306
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2718088, upper bound: 4.2715940
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2717546, upper bound: 4.2716294
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2715921
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2717546, upper bound: 4.2716294
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2715921
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716698, upper bound: 4.2716983
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716635
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716698, upper bound: 4.2716983
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716635
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716726, upper bound: 4.2716994
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2717340, upper bound: 4.2716635
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716726, upper bound: 4.2716994
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2717340, upper bound: 4.2716635
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717340
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2716726
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716635, upper bound: 4.2717335
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716983, upper bound: 4.2716698
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2715921, upper bound: 4.2718087
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2717546
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2716994, upper bound: 4.2717341
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2716984, upper bound: 4.2717335
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2716294, upper bound: 4.2718087
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2716307, upper bound: 4.2718089
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2718089, upper bound: 4.2716307
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2718087, upper bound: 4.2716294
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2717335, upper bound: 4.2716984
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 8, lower bound: -4.2717341, upper bound: 4.2716994

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 5.61 + 595.13 = 600.75 seconds
