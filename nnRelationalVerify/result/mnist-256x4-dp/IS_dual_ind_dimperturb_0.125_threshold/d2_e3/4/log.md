## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00100386


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0021743, -0.0015349, -0.0021743, -0.0015349, -0.0002945, 0.0002945)
1: (-0.0023419, -0.0004788, -0.0023419, -0.0004788, -0.0008914, 0.0008914)
2: (0.0046560, 0.0062211, 0.0046560, 0.0062211, -0.0007736, 0.0007736)
3: (-0.0041640, -0.0040137, -0.0041640, -0.0040137, -0.0000773, 0.0000773)
4: (0.0045433, 0.0058559, 0.0045433, 0.0058559, -0.0006484, 0.0006484)
5: (-0.0010512, 0.0003185, -0.0010512, 0.0003185, -0.0006895, 0.0006895)
6: (-0.0056014, -0.0048774, -0.0056014, -0.0048774, -0.0003438, 0.0003438)
7: (0.0008670, 0.0020918, 0.0008670, 0.0020918, -0.0005814, 0.0005814)
8: (-0.0004134, -0.0002404, -0.0004134, -0.0002404, -0.0000978, 0.0000978)
9: (1.0048058, 1.0081332, 1.0048058, 1.0081332, -0.0016555, 0.0016555)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.36 + 1.23 = 2.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0010733, upper bound: 0.0010733

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0010234, upper bound: 0.0007763
time: 0.41 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0010311, upper bound: 0.0010311
time: 0.41 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.97 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.97
Output dim: 9, lower bound: -0.0010234, upper bound: 0.0007763
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.97
Output dim: 9, lower bound: -0.0010311, upper bound: 0.0010311

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0021895, -0.0016042, -0.0021742, -0.0015500, -0.0002708, 0.0002030
1: -0.0020965, -0.0004436, -0.0022879, -0.0004792, -0.0005730, 0.0007877
2: 0.0048849, 0.0062469, 0.0047034, 0.0062198, -0.0004790, 0.0006694
3: -0.0041699, -0.0040312, -0.0041639, -0.0040175, -0.0000730, 0.0000550
4: 0.0045103, 0.0056745, 0.0045455, 0.0058166, -0.0005543, 0.0004035
5: -0.0008371, 0.0003503, -0.0010030, 0.0003173, -0.0004080, 0.0005865
6: -0.0055127, -0.0048628, -0.0055819, -0.0048792, -0.0002250, 0.0002959
7: 0.0008536, 0.0019465, 0.0008705, 0.0020609, -0.0005069, 0.0003956
8: -0.0003843, -0.0002323, -0.0004071, -0.0002404, -0.0000589, 0.0000866
9: 1.0047597, 1.0076038, 1.0048084, 1.0080196, -0.0014176, 0.0009897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0007763, upper bound: 0.0007763
time: 0.44 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0007763, upper bound: 0.0007763
time: 0.44 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0021743, -0.0015386, -0.0021743, -0.0015349, -0.0002945, 0.0002216
1: -0.0023330, -0.0004790, -0.0023419, -0.0004788, -0.0005932, 0.0008913
2: 0.0046656, 0.0062207, 0.0046560, 0.0062211, -0.0005051, 0.0007724
3: -0.0041640, -0.0040152, -0.0041640, -0.0040137, -0.0000773, 0.0000632
4: 0.0045440, 0.0058471, 0.0045433, 0.0058559, -0.0006471, 0.0004186
5: -0.0010432, 0.0003182, -0.0010512, 0.0003185, -0.0004220, 0.0006887
6: -0.0055965, -0.0048779, -0.0056014, -0.0048774, -0.0002356, 0.0003429
7: 0.0008681, 0.0020857, 0.0008670, 0.0020918, -0.0005792, 0.0004270
8: -0.0004115, -0.0002404, -0.0004134, -0.0002404, -0.0000642, 0.0000978
9: 1.0048066, 1.0081173, 1.0048058, 1.0081332, -0.0016534, 0.0010499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0007763, upper bound: 0.0010234
time: 0.43 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0007763, upper bound: 0.0010311
time: 0.42 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.23 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.23
Output dim: 9, lower bound: -0.0007763, upper bound: 0.0007763
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.23
Output dim: 9, lower bound: -0.0007763, upper bound: 0.0007763
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 9, lower bound: -0.0007763, upper bound: 0.0010234
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 9, lower bound: -0.0007763, upper bound: 0.0010311

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0021743, -0.0015386, -0.0021895, -0.0016042, -0.0002031, 0.0002896
1: -0.0023330, -0.0004790, -0.0020965, -0.0004436, -0.0008536, 0.0005732
2: 0.0046656, 0.0062207, 0.0048849, 0.0062469, -0.0007308, 0.0004791
3: -0.0041640, -0.0040152, -0.0041699, -0.0040312, -0.0000551, 0.0000775
4: 0.0045440, 0.0058471, 0.0045103, 0.0056745, -0.0004039, 0.0006075
5: -0.0010432, 0.0003182, -0.0008371, 0.0003503, -0.0006461, 0.0004082
6: -0.0055965, -0.0048779, -0.0055127, -0.0048628, -0.0003225, 0.0002253
7: 0.0008681, 0.0020857, 0.0008536, 0.0019465, -0.0003961, 0.0005489
8: -0.0004115, -0.0002404, -0.0003843, -0.0002323, -0.0000948, 0.0000589
9: 1.0048066, 1.0081173, 1.0047597, 1.0076038, -0.0009900, 0.0015569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0006706, upper bound: 0.0009414
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0005303, upper bound: 0.0009399
time: 0.43 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0021743, -0.0015386, -0.0021743, -0.0015386, -0.0002216, 0.0002216
1: -0.0023330, -0.0004790, -0.0023330, -0.0004790, -0.0005930, 0.0005930
2: 0.0046656, 0.0062207, 0.0046656, 0.0062207, -0.0005040, 0.0005040
3: -0.0041640, -0.0040152, -0.0041640, -0.0040152, -0.0000632, 0.0000632
4: 0.0045440, 0.0058471, 0.0045440, 0.0058471, -0.0004170, 0.0004170
5: -0.0010432, 0.0003182, -0.0010432, 0.0003182, -0.0004211, 0.0004211
6: -0.0055965, -0.0048779, -0.0055965, -0.0048779, -0.0002345, 0.0002345
7: 0.0008681, 0.0020857, 0.0008681, 0.0020857, -0.0004247, 0.0004247
8: -0.0004115, -0.0002404, -0.0004115, -0.0002404, -0.0000642, 0.0000642
9: 1.0048066, 1.0081173, 1.0048066, 1.0081173, -0.0010478, 0.0010478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0006706, upper bound: 0.0009869
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0005303, upper bound: 0.0009983
time: 0.43 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.24 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.24
Output dim: 9, lower bound: -0.0006706, upper bound: 0.0009414
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.24
Output dim: 9, lower bound: -0.0005303, upper bound: 0.0009399
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.24
Output dim: 9, lower bound: -0.0006706, upper bound: 0.0009869
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.24
Output dim: 9, lower bound: -0.0005303, upper bound: 0.0009983

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.59 + 9.95 = 12.54 seconds
