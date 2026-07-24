## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00027335


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0065973, 0.0072310, 0.0065973, 0.0072310, -0.0003753, 0.0003753)
1: (0.0008605, 0.0020879, 0.0008605, 0.0020879, -0.0007269, 0.0007269)
2: (0.0000870, 0.0099874, 0.0000870, 0.0099874, -0.0058630, 0.0058630)
3: (-0.0034046, -0.0025203, -0.0034046, -0.0025203, -0.0005237, 0.0005237)
4: (0.0048177, 0.0091079, 0.0048177, 0.0091079, -0.0025407, 0.0025407)
5: (-0.0018794, -0.0012389, -0.0018794, -0.0012389, -0.0003793, 0.0003793)
6: (0.9925253, 0.9937000, 0.9925253, 0.9937000, -0.0006956, 0.0006956)
7: (-0.0046620, 0.0031041, -0.0046620, 0.0031041, -0.0045990, 0.0045990)
8: (-0.0004722, 0.0019608, -0.0004722, 0.0019608, -0.0014408, 0.0014408)
9: (-0.0112427, -0.0063867, -0.0112427, -0.0063867, -0.0028757, 0.0028757)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.46 + 1.79 = 3.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0005342, upper bound: 0.0005342

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.ADV_EXAMPLE
time: 0.65 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.ADV_EXAMPLE
time: 0.63 seconds

## IS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (is) = 3.25 + 1.44 = 4.69 seconds
