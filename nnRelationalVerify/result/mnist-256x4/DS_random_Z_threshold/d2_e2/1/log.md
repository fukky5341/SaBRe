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
execution time: IAR + RelationalAnalysis = 0.75 + 3.30 = 4.06 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -4.2840282, upper bound: 4.2840282

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2839799, upper bound: 4.2839709
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2839709, upper bound: 4.2839799
time: 1.24 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.61 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.61
Output dim: 8, lower bound: -4.2839799, upper bound: 4.2839709
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.61
Output dim: 8, lower bound: -4.2839709, upper bound: 4.2839799

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2824876, upper bound: 4.2824873
time: 10.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2824990, upper bound: 4.2824793
time: 1.35 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2837019, upper bound: 4.2837234
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2837132, upper bound: 4.2837165
time: 1.35 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.69 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.69
Output dim: 8, lower bound: -4.2824876, upper bound: 4.2824873
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.69
Output dim: 8, lower bound: -4.2824990, upper bound: 4.2824793
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.69
Output dim: 8, lower bound: -4.2837019, upper bound: 4.2837234
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.69
Output dim: 8, lower bound: -4.2837132, upper bound: 4.2837165

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2824636, upper bound: 4.2824871
time: 6.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2824873, upper bound: 4.2824608
time: 1.25 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2476544, upper bound: 4.2476969
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2476544, upper bound: 4.2476969
time: 1.28 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1988075, upper bound: 4.1988480
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1988075, upper bound: 4.1988480
time: 1.01 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2824379, upper bound: 4.2824573
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2824379, upper bound: 4.2824573
time: 6.20 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 8.70 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 8.70
Output dim: 8, lower bound: -4.2824636, upper bound: 4.2824871
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 8.70
Output dim: 8, lower bound: -4.2824873, upper bound: 4.2824608
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 8.70
Output dim: 8, lower bound: -4.2476544, upper bound: 4.2476969
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 8.70
Output dim: 8, lower bound: -4.2476544, upper bound: 4.2476969
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 8.70
Output dim: 8, lower bound: -4.1988075, upper bound: 4.1988480
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 8.70
Output dim: 8, lower bound: -4.1988075, upper bound: 4.1988480
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 8.70
Output dim: 8, lower bound: -4.2824379, upper bound: 4.2824573
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 8.70
Output dim: 8, lower bound: -4.2824379, upper bound: 4.2824573

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2812836, upper bound: 4.2816205
time: 5.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2815596, upper bound: 4.2813049
time: 3.00 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2766937, upper bound: 4.2766787
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2766937, upper bound: 4.2766787
time: 1.47 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2357101, upper bound: 4.2357224
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2357101, upper bound: 4.2357224
time: 1.08 seconds

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2474986, upper bound: 4.2476582
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2476158, upper bound: 4.2475358
time: 1.24 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2808527, upper bound: 4.2808846
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2808526, upper bound: 4.2808846
time: 1.71 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2824214, upper bound: 4.2824571
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2824376, upper bound: 4.2824392
time: 1.73 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.85 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.85
Output dim: 8, lower bound: -4.2812836, upper bound: 4.2816205
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.85
Output dim: 8, lower bound: -4.2815596, upper bound: 4.2813049
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.85
Output dim: 8, lower bound: -4.2766937, upper bound: 4.2766787
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.85
Output dim: 8, lower bound: -4.2766937, upper bound: 4.2766787
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.85
Output dim: 8, lower bound: -4.2357101, upper bound: 4.2357224
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.85
Output dim: 8, lower bound: -4.2357101, upper bound: 4.2357224
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.85
Output dim: 8, lower bound: -4.2474986, upper bound: 4.2476582
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.85
Output dim: 8, lower bound: -4.2476158, upper bound: 4.2475358
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.85
Output dim: 8, lower bound: -4.2808527, upper bound: 4.2808846
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.85
Output dim: 8, lower bound: -4.2808526, upper bound: 4.2808846
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.85
Output dim: 8, lower bound: -4.2824214, upper bound: 4.2824571
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.85
Output dim: 8, lower bound: -4.2824376, upper bound: 4.2824392

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796833, upper bound: 4.2802282
time: 4.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2798495, upper bound: 4.2800402
time: 1.42 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2803397, upper bound: 4.2800517
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2803397, upper bound: 4.2800517
time: 1.40 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2756534, upper bound: 4.2755974
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2756534, upper bound: 4.2755974
time: 1.81 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2748270, upper bound: 4.2748142
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2748270, upper bound: 4.2748142
time: 1.40 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2022036, upper bound: 4.2022937
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2022036, upper bound: 4.2022937
time: 1.04 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2356719, upper bound: 4.2355780
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2356719, upper bound: 4.2355780
time: 0.98 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791172, upper bound: 4.2793938
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793565, upper bound: 4.2791609
time: 2.03 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791476, upper bound: 4.2793939
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793582, upper bound: 4.2791343
time: 1.74 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2806915, upper bound: 4.2807806
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2806975, upper bound: 4.2807748
time: 1.43 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2812582, upper bound: 4.2815413
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2815958, upper bound: 4.2812373
time: 1.51 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.84 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2796833, upper bound: 4.2802282
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2798495, upper bound: 4.2800402
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2803397, upper bound: 4.2800517
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2803397, upper bound: 4.2800517
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2756534, upper bound: 4.2755974
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2756534, upper bound: 4.2755974
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2748270, upper bound: 4.2748142
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2748270, upper bound: 4.2748142
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2022036, upper bound: 4.2022937
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2022036, upper bound: 4.2022937
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2356719, upper bound: 4.2355780
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2356719, upper bound: 4.2355780
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2791172, upper bound: 4.2793938
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2793565, upper bound: 4.2791609
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2791476, upper bound: 4.2793939
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2793582, upper bound: 4.2791343
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2806915, upper bound: 4.2807806
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2806975, upper bound: 4.2807748
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2812582, upper bound: 4.2815413
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -4.2815958, upper bound: 4.2812373

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1164285, upper bound: 4.1165563
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1164285, upper bound: 4.1165563
time: 1.12 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1867174, upper bound: 4.1867720
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1867174, upper bound: 4.1867720
time: 0.92 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2798900, upper bound: 4.2795958
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2799022, upper bound: 4.2795778
time: 1.57 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2799166, upper bound: 4.2799323
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2802284, upper bound: 4.2796591
time: 1.61 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2743583, upper bound: 4.2745106
time: 4.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2746152, upper bound: 4.2743191
time: 1.56 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2754831, upper bound: 4.2755925
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2756420, upper bound: 4.2754296
time: 3.23 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2741082, upper bound: 4.2740829
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2741002, upper bound: 4.2740855
time: 1.30 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2728716, upper bound: 4.2729831
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2730064, upper bound: 4.2728755
time: 2.45 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791034, upper bound: 4.2793935
time: 2.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791170, upper bound: 4.2793805
time: 2.32 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793462, upper bound: 4.2791606
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793561, upper bound: 4.2791424
time: 1.85 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2788188, upper bound: 4.2792909
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2790463, upper bound: 4.2790272
time: 1.54 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2789737, upper bound: 4.2790311
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792542, upper bound: 4.2788291
time: 2.68 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2803460, upper bound: 4.2807152
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2806255, upper bound: 4.2804561
time: 1.83 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2790981, upper bound: 4.2793067
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2792246, upper bound: 4.2791518
time: 1.41 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796240, upper bound: 4.2799249
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796240, upper bound: 4.2799231
time: 1.63 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2759320, upper bound: 4.2756739
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2759320, upper bound: 4.2756739
time: 1.27 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.27 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.1164285, upper bound: 4.1165563
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.1164285, upper bound: 4.1165563
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.1867174, upper bound: 4.1867720
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.1867174, upper bound: 4.1867720
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2798900, upper bound: 4.2795958
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2799022, upper bound: 4.2795778
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2799166, upper bound: 4.2799323
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2802284, upper bound: 4.2796591
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2743583, upper bound: 4.2745106
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2746152, upper bound: 4.2743191
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2754831, upper bound: 4.2755925
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2756420, upper bound: 4.2754296
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2741082, upper bound: 4.2740829
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2741002, upper bound: 4.2740855
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2728716, upper bound: 4.2729831
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2730064, upper bound: 4.2728755
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2791034, upper bound: 4.2793935
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2791170, upper bound: 4.2793805
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2793462, upper bound: 4.2791606
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2793561, upper bound: 4.2791424
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2788188, upper bound: 4.2792909
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2790463, upper bound: 4.2790272
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2789737, upper bound: 4.2790311
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2792542, upper bound: 4.2788291
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2803460, upper bound: 4.2807152
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2806255, upper bound: 4.2804561
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2790981, upper bound: 4.2793067
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2792246, upper bound: 4.2791518
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2796240, upper bound: 4.2799249
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2796240, upper bound: 4.2799231
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2759320, upper bound: 4.2756739
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -4.2759320, upper bound: 4.2756739

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2738797, upper bound: 4.2736103
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2738797, upper bound: 4.2736103
time: 2.99 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.9919806, upper bound: 3.9920613
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.9919806, upper bound: 3.9920613
time: 1.02 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2785419, upper bound: 4.2785132
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2785419, upper bound: 4.2785132
time: 1.45 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.0296154, upper bound: 4.0297239
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.0296154, upper bound: 4.0297239
time: 0.92 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2721228, upper bound: 4.2722383
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2721228, upper bound: 4.2722383
time: 2.68 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2724681, upper bound: 4.2719859
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2724681, upper bound: 4.2719859
time: 1.28 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2736444, upper bound: 4.2737836
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2737346, upper bound: 4.2737224
time: 1.48 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2731391, upper bound: 4.2728488
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2731391, upper bound: 4.2728488
time: 1.21 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2732151, upper bound: 4.2733352
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2733876, upper bound: 4.2732103
time: 1.24 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2732102, upper bound: 4.2733384
time: 4.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2733780, upper bound: 4.2732109
time: 1.56 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2720426, upper bound: 4.2721359
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2720348, upper bound: 4.2721447
time: 1.23 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2711041, upper bound: 4.2708670
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2711041, upper bound: 4.2708670
time: 2.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2787917, upper bound: 4.2792906
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2789997, upper bound: 4.2790034
time: 1.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2788186, upper bound: 4.2792738
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2790146, upper bound: 4.2789721
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2729532, upper bound: 4.2729424
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2729532, upper bound: 4.2729424
time: 1.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.0266088, upper bound: 4.0265930
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.0266088, upper bound: 4.0265930
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.0261460, upper bound: 4.0261981
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.0261460, upper bound: 4.0261981
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2777411, upper bound: 4.2779729
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2780762, upper bound: 4.2776081
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2727885, upper bound: 4.2729448
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2727885, upper bound: 4.2729448
time: 1.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2778712, upper bound: 4.2777810
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2782346, upper bound: 4.2774684
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2790529, upper bound: 4.2793688
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2790448, upper bound: 4.2793679
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794972, upper bound: 4.2796560
time: 4.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2798142, upper bound: 4.2793241
time: 2.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2781289, upper bound: 4.2783935
time: 3.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2784402, upper bound: 4.2780801
time: 1.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2728563, upper bound: 4.2731381
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2728563, upper bound: 4.2731381
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2780774, upper bound: 4.2784961
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2782001, upper bound: 4.2783285
time: 1.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2731346, upper bound: 4.2728547
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2731346, upper bound: 4.2728547
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2740142, upper bound: 4.2737430
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2740559, upper bound: 4.2737106
time: 1.53 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.60 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2738797, upper bound: 4.2736103
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2738797, upper bound: 4.2736103
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -3.9919806, upper bound: 3.9920613
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -3.9919806, upper bound: 3.9920613
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2785419, upper bound: 4.2785132
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2785419, upper bound: 4.2785132
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.0296154, upper bound: 4.0297239
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.0296154, upper bound: 4.0297239
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2721228, upper bound: 4.2722383
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2721228, upper bound: 4.2722383
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2724681, upper bound: 4.2719859
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2724681, upper bound: 4.2719859
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2736444, upper bound: 4.2737836
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2737346, upper bound: 4.2737224
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2731391, upper bound: 4.2728488
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2731391, upper bound: 4.2728488
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2732151, upper bound: 4.2733352
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2733876, upper bound: 4.2732103
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2732102, upper bound: 4.2733384
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2733780, upper bound: 4.2732109
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2720426, upper bound: 4.2721359
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2720348, upper bound: 4.2721447
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2711041, upper bound: 4.2708670
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2711041, upper bound: 4.2708670
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2787917, upper bound: 4.2792906
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2789997, upper bound: 4.2790034
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2788186, upper bound: 4.2792738
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2790146, upper bound: 4.2789721
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2729532, upper bound: 4.2729424
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2729532, upper bound: 4.2729424
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.0266088, upper bound: 4.0265930
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.0266088, upper bound: 4.0265930
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.0261460, upper bound: 4.0261981
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.0261460, upper bound: 4.0261981
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2777411, upper bound: 4.2779729
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2780762, upper bound: 4.2776081
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2727885, upper bound: 4.2729448
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2727885, upper bound: 4.2729448
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2778712, upper bound: 4.2777810
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2782346, upper bound: 4.2774684
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2790529, upper bound: 4.2793688
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2790448, upper bound: 4.2793679
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2794972, upper bound: 4.2796560
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2798142, upper bound: 4.2793241
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2718088
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2781289, upper bound: 4.2783935
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2784402, upper bound: 4.2780801
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2728563, upper bound: 4.2731381
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2728563, upper bound: 4.2731381
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2780774, upper bound: 4.2784961
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2782001, upper bound: 4.2783285
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2731346, upper bound: 4.2728547
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2731346, upper bound: 4.2728547
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2740142, upper bound: 4.2737430
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -4.2740559, upper bound: 4.2737106

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715615, upper bound: 4.2712920
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715615, upper bound: 4.2712920
time: 2.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718856, upper bound: 4.2716981
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2719545, upper bound: 4.2716581
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2769221, upper bound: 4.2770319
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2770684, upper bound: 4.2769175
time: 2.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2721472, upper bound: 4.2720546
time: 4.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2721472, upper bound: 4.2720546
time: 1.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2699488, upper bound: 4.2701441
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2700100, upper bound: 4.2700844
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2699488, upper bound: 4.2701441
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2700100, upper bound: 4.2700844
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2722398, upper bound: 4.2719672
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2724619, upper bound: 4.2718195
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2722398, upper bound: 4.2719672
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2724619, upper bound: 4.2718195
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2728253, upper bound: 4.2729433
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2728164, upper bound: 4.2729432
time: 1.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2723214, upper bound: 4.2725686
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2725911, upper bound: 4.2723610
time: 2.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2709927, upper bound: 4.2707885
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2711041, upper bound: 4.2707281
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2723898, upper bound: 4.2721305
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2723877, upper bound: 4.2721207
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2729790, upper bound: 4.2733027
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2731447, upper bound: 4.2731672
time: 1.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2732150, upper bound: 4.2731516
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2733668, upper bound: 4.2729715
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2711383, upper bound: 4.2713397
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2712161, upper bound: 4.2712730
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2732041, upper bound: 4.2731609
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2733417, upper bound: 4.2729750
time: 2.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2719253, upper bound: 4.2721359
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2720426, upper bound: 4.2720235
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2702396, upper bound: 4.2701911
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2702396, upper bound: 4.2701911
time: 1.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2703367, upper bound: 4.2701225
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2703364, upper bound: 4.2701180
time: 2.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2703367, upper bound: 4.2701225
time: 2.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2703364, upper bound: 4.2701180
time: 1.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2774470, upper bound: 4.2779002
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2774470, upper bound: 4.2779002
time: 2.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2728854, upper bound: 4.2728335
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2728854, upper bound: 4.2728335
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2774860, upper bound: 4.2778599
time: 3.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2774860, upper bound: 4.2778599
time: 3.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.0261475, upper bound: 4.0260816
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.0261475, upper bound: 4.0260816
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716890, upper bound: 4.2719589
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2719033, upper bound: 4.2716878
time: 1.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2728035, upper bound: 4.2729424
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2729532, upper bound: 4.2727931
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2765559, upper bound: 4.2767952
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2765559, upper bound: 4.2767952
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718976, upper bound: 4.2715884
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718976, upper bound: 4.2715884
time: 2.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2700916, upper bound: 4.2702405
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2700916, upper bound: 4.2702405
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2727778, upper bound: 4.2729442
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2727879, upper bound: 4.2729185
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.9698597, upper bound: 3.9698647
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.9698597, upper bound: 3.9698647
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.9698597, upper bound: 3.9698647
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.9698597, upper bound: 3.9698647
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2721363, upper bound: 4.2723899
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2721363, upper bound: 4.2723899
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2721305, upper bound: 4.2723898
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2721305, upper bound: 4.2723898
time: 1.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2779565, upper bound: 4.2782214
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2780595, upper bound: 4.2780742
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2730276, upper bound: 4.2727449
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2730276, upper bound: 4.2727449
time: 1.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2714687, upper bound: 4.2718088
time: 3.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2716237
time: 2.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2706767, upper bound: 4.2711670
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2708848, upper bound: 4.2708155
time: 1.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2778139, upper bound: 4.2783258
time: 2.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2780595, upper bound: 4.2780742
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2709540, upper bound: 4.2707587
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2709540, upper bound: 4.2707587
time: 1.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2707103, upper bound: 4.2710511
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2707645, upper bound: 4.2709569
time: 1.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2727265, upper bound: 4.2731381
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2728563, upper bound: 4.2729486
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2766128, upper bound: 4.2770885
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2766450, upper bound: 4.2770868
time: 1.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2707645, upper bound: 4.2709569
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2707645, upper bound: 4.2709569
time: 3.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2709769, upper bound: 4.2707669
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2710665, upper bound: 4.2707240
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2709769, upper bound: 4.2707669
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2710665, upper bound: 4.2707240
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2709769, upper bound: 4.2707669
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2709769, upper bound: 4.2707669
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2738572, upper bound: 4.2736896
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2740553, upper bound: 4.2735346
time: 2.20 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 4.34 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2715615, upper bound: 4.2712920
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2715615, upper bound: 4.2712920
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2718856, upper bound: 4.2716981
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2719545, upper bound: 4.2716581
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2769221, upper bound: 4.2770319
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2770684, upper bound: 4.2769175
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2721472, upper bound: 4.2720546
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2721472, upper bound: 4.2720546
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2699488, upper bound: 4.2701441
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2700100, upper bound: 4.2700844
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2699488, upper bound: 4.2701441
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2700100, upper bound: 4.2700844
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2722398, upper bound: 4.2719672
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2724619, upper bound: 4.2718195
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2722398, upper bound: 4.2719672
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2724619, upper bound: 4.2718195
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2728253, upper bound: 4.2729433
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2728164, upper bound: 4.2729432
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2723214, upper bound: 4.2725686
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2725911, upper bound: 4.2723610
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2709927, upper bound: 4.2707885
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2711041, upper bound: 4.2707281
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2723898, upper bound: 4.2721305
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2723877, upper bound: 4.2721207
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2729790, upper bound: 4.2733027
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2731447, upper bound: 4.2731672
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2732150, upper bound: 4.2731516
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2733668, upper bound: 4.2729715
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2711383, upper bound: 4.2713397
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2712161, upper bound: 4.2712730
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2732041, upper bound: 4.2731609
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2733417, upper bound: 4.2729750
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2719253, upper bound: 4.2721359
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2720426, upper bound: 4.2720235
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2702396, upper bound: 4.2701911
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2702396, upper bound: 4.2701911
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2703367, upper bound: 4.2701225
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2703364, upper bound: 4.2701180
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2703367, upper bound: 4.2701225
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2703364, upper bound: 4.2701180
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2774470, upper bound: 4.2779002
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2774470, upper bound: 4.2779002
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2728854, upper bound: 4.2728335
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2728854, upper bound: 4.2728335
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2774860, upper bound: 4.2778599
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2774860, upper bound: 4.2778599
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.0261475, upper bound: 4.0260816
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.0261475, upper bound: 4.0260816
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2716890, upper bound: 4.2719589
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2719033, upper bound: 4.2716878
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2728035, upper bound: 4.2729424
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2729532, upper bound: 4.2727931
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2765559, upper bound: 4.2767952
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2765559, upper bound: 4.2767952
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2718976, upper bound: 4.2715884
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2718976, upper bound: 4.2715884
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2700916, upper bound: 4.2702405
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2700916, upper bound: 4.2702405
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2727778, upper bound: 4.2729442
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2727879, upper bound: 4.2729185
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.34
Output dim: 8, lower bound: -3.9698597, upper bound: 3.9698647
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.34
Output dim: 8, lower bound: -3.9698597, upper bound: 3.9698647
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.34
Output dim: 8, lower bound: -3.9698597, upper bound: 3.9698647
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.34
Output dim: 8, lower bound: -3.9698597, upper bound: 3.9698647
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2721363, upper bound: 4.2723899
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2721363, upper bound: 4.2723899
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2721305, upper bound: 4.2723898
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2721305, upper bound: 4.2723898
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2779565, upper bound: 4.2782214
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2780595, upper bound: 4.2780742
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2730276, upper bound: 4.2727449
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2730276, upper bound: 4.2727449
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2714687, upper bound: 4.2718088
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2716237
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2706767, upper bound: 4.2711670
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2708848, upper bound: 4.2708155
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2778139, upper bound: 4.2783258
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2780595, upper bound: 4.2780742
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2709540, upper bound: 4.2707587
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2709540, upper bound: 4.2707587
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2707103, upper bound: 4.2710511
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2707645, upper bound: 4.2709569
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2727265, upper bound: 4.2731381
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2728563, upper bound: 4.2729486
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2766128, upper bound: 4.2770885
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2766450, upper bound: 4.2770868
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2707645, upper bound: 4.2709569
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2707645, upper bound: 4.2709569
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2709769, upper bound: 4.2707669
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2710665, upper bound: 4.2707240
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2709769, upper bound: 4.2707669
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2710665, upper bound: 4.2707240
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2709769, upper bound: 4.2707669
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2709769, upper bound: 4.2707669
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2738572, upper bound: 4.2736896
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.34
Output dim: 8, lower bound: -4.2740553, upper bound: 4.2735346

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2713763, upper bound: 4.2712920
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715615, upper bound: 4.2711267
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2713763, upper bound: 4.2712920
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715615, upper bound: 4.2711267
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2717421, upper bound: 4.2716820
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718651, upper bound: 4.2715359
time: 1.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2693992, upper bound: 4.2690860
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2693992, upper bound: 4.2690860
time: 1.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2699942, upper bound: 4.2699548
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2699942, upper bound: 4.2699548
time: 3.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2700612, upper bound: 4.2699047
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2700612, upper bound: 4.2699047
time: 1.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2713763, upper bound: 4.2712920
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2713897, upper bound: 4.2712817
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2699942, upper bound: 4.2699548
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2700612, upper bound: 4.2699047
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2691280, upper bound: 4.2692983
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2691290, upper bound: 4.2692808
time: 1.46 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 3.51 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 8, lower bound: -4.2713763, upper bound: 4.2712920
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 8, lower bound: -4.2715615, upper bound: 4.2711267
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 8, lower bound: -4.2713763, upper bound: 4.2712920
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 8, lower bound: -4.2715615, upper bound: 4.2711267
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 8, lower bound: -4.2717421, upper bound: 4.2716820
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 8, lower bound: -4.2718651, upper bound: 4.2715359
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 8, lower bound: -4.2693992, upper bound: 4.2690860
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 8, lower bound: -4.2693992, upper bound: 4.2690860
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 8, lower bound: -4.2699942, upper bound: 4.2699548
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 8, lower bound: -4.2699942, upper bound: 4.2699548
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 8, lower bound: -4.2700612, upper bound: 4.2699047
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 8, lower bound: -4.2700612, upper bound: 4.2699047
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 8, lower bound: -4.2713763, upper bound: 4.2712920
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 8, lower bound: -4.2713897, upper bound: 4.2712817
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 8, lower bound: -4.2699942, upper bound: 4.2699548
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 8, lower bound: -4.2700612, upper bound: 4.2699047
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 8, lower bound: -4.2691280, upper bound: 4.2692983
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 8, lower bound: -4.2691290, upper bound: 4.2692808
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2700100, upper bound: 4.2700844
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2699488, upper bound: 4.2701441
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2700100, upper bound: 4.2700844
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2722398, upper bound: 4.2719672
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2724619, upper bound: 4.2718195
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2722398, upper bound: 4.2719672
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2724619, upper bound: 4.2718195
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2728253, upper bound: 4.2729433
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2728164, upper bound: 4.2729432
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2723214, upper bound: 4.2725686
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2725911, upper bound: 4.2723610
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2709927, upper bound: 4.2707885
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2711041, upper bound: 4.2707281
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2723898, upper bound: 4.2721305
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2723877, upper bound: 4.2721207
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2729790, upper bound: 4.2733027
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2731447, upper bound: 4.2731672
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2732150, upper bound: 4.2731516
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2733668, upper bound: 4.2729715
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2711383, upper bound: 4.2713397
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2712161, upper bound: 4.2712730
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2732041, upper bound: 4.2731609
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2733417, upper bound: 4.2729750
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2719253, upper bound: 4.2721359
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2720426, upper bound: 4.2720235
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2702396, upper bound: 4.2701911
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2702396, upper bound: 4.2701911
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2703367, upper bound: 4.2701225
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2703364, upper bound: 4.2701180
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2703367, upper bound: 4.2701225
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2703364, upper bound: 4.2701180
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2774470, upper bound: 4.2779002
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2774470, upper bound: 4.2779002
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2728854, upper bound: 4.2728335
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2728854, upper bound: 4.2728335
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2774860, upper bound: 4.2778599
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2774860, upper bound: 4.2778599
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2716890, upper bound: 4.2719589
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2719033, upper bound: 4.2716878
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2728035, upper bound: 4.2729424
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2729532, upper bound: 4.2727931
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2765559, upper bound: 4.2767952
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2765559, upper bound: 4.2767952
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2718976, upper bound: 4.2715884
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2718976, upper bound: 4.2715884
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2700916, upper bound: 4.2702405
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2700916, upper bound: 4.2702405
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2727778, upper bound: 4.2729442
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2727879, upper bound: 4.2729185
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2721363, upper bound: 4.2723899
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2721363, upper bound: 4.2723899
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2721305, upper bound: 4.2723898
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2721305, upper bound: 4.2723898
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2779565, upper bound: 4.2782214
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2780595, upper bound: 4.2780742
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2730276, upper bound: 4.2727449
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2730276, upper bound: 4.2727449
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2714687, upper bound: 4.2718088
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2715940, upper bound: 4.2716237
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2706767, upper bound: 4.2711670
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2708848, upper bound: 4.2708155
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2778139, upper bound: 4.2783258
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2780595, upper bound: 4.2780742
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2709540, upper bound: 4.2707587
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2709540, upper bound: 4.2707587
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2707103, upper bound: 4.2710511
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2707645, upper bound: 4.2709569
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2727265, upper bound: 4.2731381
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2728563, upper bound: 4.2729486
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2766128, upper bound: 4.2770885
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2766450, upper bound: 4.2770868
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2707645, upper bound: 4.2709569
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2707645, upper bound: 4.2709569
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2709769, upper bound: 4.2707669
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2710665, upper bound: 4.2707240
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2709769, upper bound: 4.2707669
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2710665, upper bound: 4.2707240
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2709769, upper bound: 4.2707669
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2709769, upper bound: 4.2707669
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2738572, upper bound: 4.2736896
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 8, lower bound: -4.2740553, upper bound: 4.2735346

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.06 + 596.59 = 600.65 seconds
