## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0028484


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0027474, -0.0022608, -0.0027474, -0.0022608, -0.0002299, 0.0002299)
1: (0.0241357, 0.0270717, 0.0241357, 0.0270717, -0.0013117, 0.0013117)
2: (0.0235094, 0.0254717, 0.0235094, 0.0254717, -0.0008935, 0.0008935)
3: (0.0113954, 0.0135621, 0.0113954, 0.0135621, -0.0010990, 0.0010990)
4: (-0.0137747, -0.0114777, -0.0137747, -0.0114777, -0.0011752, 0.0011752)
5: (0.0187185, 0.0213942, 0.0187185, 0.0213942, -0.0013777, 0.0013777)
6: (0.0092998, 0.0113680, 0.0092998, 0.0113680, -0.0010703, 0.0010703)
7: (-0.0185286, -0.0164152, -0.0185286, -0.0164152, -0.0010181, 0.0010181)
8: (0.0133614, 0.0154880, 0.0133614, 0.0154880, -0.0010918, 0.0010918)
9: (0.9182836, 0.9286151, 0.9182836, 0.9286151, -0.0051571, 0.0051571)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.38 + 1.21 = 2.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0037108, upper bound: 0.0037108

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 134

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.ADV_EXAMPLE
time: 0.42 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.ADV_EXAMPLE
time: 0.36 seconds

## IS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (is) = 2.59 + 0.92 = 3.51 seconds
