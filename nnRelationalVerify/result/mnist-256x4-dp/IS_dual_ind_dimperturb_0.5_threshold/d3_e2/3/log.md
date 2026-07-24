## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.37300728


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.2403145, 0.1862360, -0.2403145, 0.1862360, -0.4265504, 0.4265504)
1: (-0.2206881, 0.1867502, -0.2206881, 0.1867502, -0.4074384, 0.4074384)
2: (-0.1603214, 0.2868525, -0.1603214, 0.2868525, -0.4471739, 0.4471739)
3: (-0.1200541, 0.3555790, -0.1200541, 0.3555790, -0.4709151, 0.4709151)
4: (-0.1951303, 0.2452313, -0.1951303, 0.2452313, -0.4403616, 0.4403616)
5: (-0.1922414, 0.2739196, -0.1922414, 0.2739196, -0.4661610, 0.4661610)
6: (-0.2326098, 0.2268756, -0.2326098, 0.2268756, -0.4594854, 0.4594854)
7: (0.4701264, 1.0939515, 0.4701264, 1.0939515, -0.6238251, 0.6238251)
8: (-0.1828370, 0.3089072, -0.1828370, 0.3089072, -0.4917443, 0.4917443)
9: (-0.1807162, 0.2984259, -0.1807162, 0.2984259, -0.4791422, 0.4791422)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.52 + 2.12 = 3.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.5035486, upper bound: 0.5035486

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4819067, upper bound: 0.4855903
time: 1.00 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4789193, upper bound: 0.4789193
time: 0.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.10 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.10
Output dim: 7, lower bound: -0.4819067, upper bound: 0.4855903
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.10
Output dim: 7, lower bound: -0.4789193, upper bound: 0.4789193

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.2251547, 0.1675636, -0.2403145, 0.1862360, -0.4113907, 0.4078780
1: -0.2059905, 0.1711420, -0.2206881, 0.1867502, -0.3927408, 0.3918301
2: -0.1460187, 0.2697790, -0.1603214, 0.2868525, -0.4328713, 0.4301004
3: -0.1151546, 0.3279528, -0.1200541, 0.3555790, -0.4656529, 0.4434546
4: -0.1823285, 0.2247390, -0.1951303, 0.2452313, -0.4275598, 0.4198692
5: -0.1762433, 0.2547885, -0.1922414, 0.2739196, -0.4501629, 0.4470299
6: -0.2147308, 0.2088974, -0.2326098, 0.2268756, -0.4416064, 0.4415072
7: 0.5055090, 1.0899537, 0.4701264, 1.0939515, -0.5884424, 0.6198273
8: -0.1653667, 0.2920361, -0.1828370, 0.3089072, -0.4742739, 0.4748732
9: -0.1655571, 0.2826715, -0.1807162, 0.2984259, -0.4639830, 0.4633878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4789193, upper bound: 0.4789193
time: 0.92 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4789193, upper bound: 0.4789193
time: 0.93 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.2091263, 0.1503158, -0.2353465, 0.1800589, -0.3891852, 0.3856624
1: -0.1925352, 0.1568974, -0.2159944, 0.1819204, -0.3744556, 0.3728918
2: -0.1334668, 0.2535309, -0.1558203, 0.2812782, -0.4147449, 0.4093512
3: -0.1248073, 0.3059977, -0.1189493, 0.3477572, -0.4667280, 0.4205402
4: -0.1691680, 0.2036187, -0.1909830, 0.2385207, -0.4076886, 0.3946017
5: -0.1602119, 0.2361802, -0.1871172, 0.2675898, -0.4278017, 0.4232973
6: -0.1972268, 0.1912714, -0.2267869, 0.2210447, -0.4182715, 0.4180583
7: 0.5334520, 1.1101909, 0.4802887, 1.0930636, -0.5596116, 0.6299022
8: -0.1476488, 0.2753505, -0.1771059, 0.3033729, -0.4510218, 0.4524564
9: -0.1523299, 0.2667730, -0.1758776, 0.2932968, -0.4456267, 0.4426506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4789193, upper bound: 0.4789193
time: 1.02 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4789193, upper bound: 0.4789193
time: 1.00 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.52 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 7, lower bound: -0.4789193, upper bound: 0.4789193
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 7, lower bound: -0.4789193, upper bound: 0.4789193
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 7, lower bound: -0.4789193, upper bound: 0.4789193
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 7, lower bound: -0.4789193, upper bound: 0.4789193

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.2251547, 0.1675636, -0.2251547, 0.1675636, -0.3927183, 0.3927183
1: -0.2059905, 0.1711420, -0.2059905, 0.1711420, -0.3771325, 0.3771325
2: -0.1460187, 0.2697790, -0.1460187, 0.2697790, -0.4157977, 0.4157977
3: -0.1151546, 0.3279528, -0.1151546, 0.3279528, -0.4381924, 0.4381924
4: -0.1823285, 0.2247390, -0.1823285, 0.2247390, -0.4070674, 0.4070674
5: -0.1762433, 0.2547885, -0.1762433, 0.2547885, -0.4310318, 0.4310318
6: -0.2147308, 0.2088974, -0.2147308, 0.2088974, -0.4236283, 0.4236283
7: 0.5055090, 1.0899537, 0.5055090, 1.0899537, -0.5844446, 0.5844446
8: -0.1653667, 0.2920361, -0.1653667, 0.2920361, -0.4574028, 0.4574028
9: -0.1655571, 0.2826715, -0.1655571, 0.2826715, -0.4482286, 0.4482286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4792668, upper bound: 0.4813088
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4792668, upper bound: 0.4844610
time: 0.95 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.2251547, 0.1675636, -0.2091263, 0.1503158, -0.3754705, 0.3766899
1: -0.2059905, 0.1711420, -0.1925352, 0.1568974, -0.3628879, 0.3636772
2: -0.1460187, 0.2697790, -0.1334668, 0.2535309, -0.3995496, 0.4032457
3: -0.1151546, 0.3279528, -0.1248073, 0.3059977, -0.4148839, 0.4471180
4: -0.1823285, 0.2247390, -0.1691680, 0.2036187, -0.3859472, 0.3939070
5: -0.1762433, 0.2547885, -0.1602119, 0.2361802, -0.4124234, 0.4150004
6: -0.2147308, 0.2088974, -0.1972268, 0.1912714, -0.4060023, 0.4061242
7: 0.5055090, 1.0899537, 0.5334520, 1.1101909, -0.6046818, 0.5565016
8: -0.1653667, 0.2920361, -0.1476488, 0.2753505, -0.4407172, 0.4396850
9: -0.1655571, 0.2826715, -0.1523299, 0.2667730, -0.4323301, 0.4350014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4792668, upper bound: 0.4813088
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4792668, upper bound: 0.4844610
time: 1.07 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.2091263, 0.1503158, -0.2251547, 0.1675636, -0.3766899, 0.3754705
1: -0.1925352, 0.1568974, -0.2059905, 0.1711420, -0.3636772, 0.3628879
2: -0.1334668, 0.2535309, -0.1460187, 0.2697790, -0.4032457, 0.3995496
3: -0.1248073, 0.3059977, -0.1151546, 0.3279528, -0.4471180, 0.4148837
4: -0.1691680, 0.2036187, -0.1823285, 0.2247390, -0.3939070, 0.3859472
5: -0.1602119, 0.2361802, -0.1762433, 0.2547885, -0.4150004, 0.4124234
6: -0.1972268, 0.1912714, -0.2147308, 0.2088974, -0.4061242, 0.4060023
7: 0.5334520, 1.1101909, 0.5055090, 1.0899537, -0.5565016, 0.6046818
8: -0.1476488, 0.2753505, -0.1653667, 0.2920361, -0.4396850, 0.4407172
9: -0.1523299, 0.2667730, -0.1655571, 0.2826715, -0.4350014, 0.4323301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4768866, upper bound: 0.4740589
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4768866, upper bound: 0.4768866
time: 1.13 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.2091263, 0.1503158, -0.2091263, 0.1503158, -0.3594422, 0.3594422
1: -0.1925352, 0.1568974, -0.1925352, 0.1568974, -0.3494326, 0.3494326
2: -0.1334668, 0.2535309, -0.1334668, 0.2535309, -0.3869976, 0.3869976
3: -0.1248073, 0.3059977, -0.1248073, 0.3059977, -0.4230915, 0.4230916
4: -0.1691680, 0.2036187, -0.1691680, 0.2036187, -0.3727867, 0.3727867
5: -0.1602119, 0.2361802, -0.1602119, 0.2361802, -0.3963921, 0.3963921
6: -0.1972268, 0.1912714, -0.1972268, 0.1912714, -0.3884982, 0.3884982
7: 0.5334520, 1.1101909, 0.5334520, 1.1101909, -0.5767388, 0.5767388
8: -0.1476488, 0.2753505, -0.1476488, 0.2753505, -0.4229994, 0.4229994
9: -0.1523299, 0.2667730, -0.1523299, 0.2667730, -0.4191029, 0.4191029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4768866, upper bound: 0.4740589
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4768866, upper bound: 0.4768866
time: 1.04 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.64 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 7, lower bound: -0.4792668, upper bound: 0.4813088
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 7, lower bound: -0.4792668, upper bound: 0.4844610
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 7, lower bound: -0.4792668, upper bound: 0.4813088
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 7, lower bound: -0.4792668, upper bound: 0.4844610
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 7, lower bound: -0.4768866, upper bound: 0.4740589
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 7, lower bound: -0.4768866, upper bound: 0.4768866
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 7, lower bound: -0.4768866, upper bound: 0.4740589
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 7, lower bound: -0.4768866, upper bound: 0.4768866

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2183168, 0.1587714, -0.2240478, 0.1661556, -0.3844724, 0.3828192
1: -0.1994922, 0.1648549, -0.2049429, 0.1701061, -0.3695983, 0.3697977
2: -0.1391376, 0.2617632, -0.1449424, 0.2685003, -0.4076379, 0.4067056
3: -0.1044437, 0.3201911, -0.1138408, 0.3265586, -0.4262714, 0.4279723
4: -0.1766559, 0.2155402, -0.1814091, 0.2232475, -0.3999034, 0.3969493
5: -0.1693854, 0.2460482, -0.1751241, 0.2533752, -0.4227606, 0.4211723
6: -0.2068092, 0.2005898, -0.2134434, 0.2075691, -0.4143783, 0.4140332
7: 0.5165811, 1.0787573, 0.5074274, 1.0886202, -0.5720391, 0.5713299
8: -0.1577146, 0.2836863, -0.1641157, 0.2907229, -0.4484375, 0.4478020
9: -0.1583537, 0.2754436, -0.1644197, 0.2815119, -0.4398656, 0.4398633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4902791, upper bound: 0.4902791
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4902791, upper bound: 0.4902791
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2206964, 0.1619159, -0.2251547, 0.1675636, -0.3882601, 0.3870706
1: -0.2017967, 0.1670093, -0.2059905, 0.1711420, -0.3729387, 0.3729998
2: -0.1417449, 0.2646741, -0.1460187, 0.2697790, -0.4115238, 0.4106928
3: -0.1102066, 0.3223836, -0.1151546, 0.3279528, -0.4331146, 0.4314207
4: -0.1786315, 0.2187539, -0.1823285, 0.2247390, -0.4033705, 0.4010824
5: -0.1717529, 0.2491017, -0.1762433, 0.2547885, -0.4265414, 0.4253449
6: -0.2095469, 0.2035981, -0.2147308, 0.2088974, -0.4184443, 0.4183289
7: 0.5131555, 1.0853381, 0.5055090, 1.0899537, -0.5767982, 0.5798291
8: -0.1603319, 0.2867908, -0.1653667, 0.2920361, -0.4523680, 0.4521576
9: -0.1610460, 0.2780313, -0.1655571, 0.2826715, -0.4437175, 0.4435884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4902791, upper bound: 0.4952743
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4902791, upper bound: 0.4953219
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2183168, 0.1587714, -0.2080763, 0.1493613, -0.3676781, 0.3668477
1: -0.1994922, 0.1648549, -0.1916938, 0.1559179, -0.3554100, 0.3565487
2: -0.1391376, 0.2617632, -0.1324517, 0.2524495, -0.3915871, 0.3942149
3: -0.1044437, 0.3201911, -0.1234874, 0.3046834, -0.4030473, 0.4368815
4: -0.1766559, 0.2155402, -0.1682969, 0.2022310, -0.3788869, 0.3838371
5: -0.1693854, 0.2460482, -0.1591532, 0.2351366, -0.4045220, 0.4052014
6: -0.2068092, 0.2005898, -0.1962010, 0.1900162, -0.3968254, 0.3967909
7: 0.5165811, 1.0787573, 0.5351154, 1.1088398, -0.5922587, 0.5436419
8: -0.1577146, 0.2836863, -0.1466027, 0.2741075, -0.4318221, 0.4302890
9: -0.1583537, 0.2754436, -0.1513267, 0.2656769, -0.4240305, 0.4267703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4769800, upper bound: 0.4813088
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4769800, upper bound: 0.4813088
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2206964, 0.1619159, -0.2091263, 0.1503158, -0.3710123, 0.3710422
1: -0.2017967, 0.1670093, -0.1925352, 0.1568974, -0.3586941, 0.3595445
2: -0.1417449, 0.2646741, -0.1334668, 0.2535309, -0.3952757, 0.3981408
3: -0.1102066, 0.3223836, -0.1248073, 0.3059977, -0.4097626, 0.4403596
4: -0.1786315, 0.2187539, -0.1691680, 0.2036187, -0.3822502, 0.3879219
5: -0.1717529, 0.2491017, -0.1602119, 0.2361802, -0.4079331, 0.4093136
6: -0.2095469, 0.2035981, -0.1972268, 0.1912714, -0.4008183, 0.4008248
7: 0.5131555, 1.0853381, 0.5334520, 1.1101909, -0.5970354, 0.5518861
8: -0.1603319, 0.2867908, -0.1476488, 0.2753505, -0.4356824, 0.4344397
9: -0.1610460, 0.2780313, -0.1523299, 0.2667730, -0.4278190, 0.4303612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4769800, upper bound: 0.4844611
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4769800, upper bound: 0.4844611
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2020375, 0.1438168, -0.2240478, 0.1661556, -0.3681930, 0.3678646
1: -0.1867528, 0.1502049, -0.2049429, 0.1701061, -0.3568589, 0.3551477
2: -0.1263626, 0.2461205, -0.1449424, 0.2685003, -0.3948628, 0.3910629
3: -0.1135045, 0.2968153, -0.1138408, 0.3265586, -0.4344630, 0.4042583
4: -0.1632741, 0.1942582, -0.1814091, 0.2232475, -0.3865216, 0.3756673
5: -0.1530318, 0.2291611, -0.1751241, 0.2533752, -0.4064070, 0.4042852
6: -0.1902892, 0.1826943, -0.2134434, 0.2075691, -0.3978582, 0.3961377
7: 0.5451492, 1.0985003, 0.5074274, 1.0886202, -0.5434710, 0.5910729
8: -0.1406359, 0.2667988, -0.1641157, 0.2907229, -0.4313588, 0.4309145
9: -0.1453531, 0.2593255, -0.1644197, 0.2815119, -0.4268650, 0.4237451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4813088, upper bound: 0.4769800
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4813088, upper bound: 0.4769800
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2047805, 0.1463989, -0.2251547, 0.1675636, -0.3723441, 0.3715536
1: -0.1890939, 0.1528869, -0.2059905, 0.1711420, -0.3602359, 0.3588774
2: -0.1293605, 0.2491073, -0.1460187, 0.2697790, -0.3991395, 0.3951260
3: -0.1199862, 0.3006130, -0.1151546, 0.3279528, -0.4421371, 0.4095144
4: -0.1655703, 0.1978955, -0.1823285, 0.2247390, -0.3903093, 0.3802239
5: -0.1558489, 0.2318737, -0.1762433, 0.2547885, -0.4106374, 0.4081169
6: -0.1929862, 0.1861362, -0.2147308, 0.2088974, -0.4018836, 0.4008670
7: 0.5402454, 1.1056819, 0.5055090, 1.0899537, -0.5497082, 0.6001729
8: -0.1433200, 0.2702668, -0.1653667, 0.2920361, -0.4353561, 0.4356335
9: -0.1482660, 0.2622738, -0.1655571, 0.2826715, -0.4309375, 0.4278310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4813088, upper bound: 0.4792668
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4813088, upper bound: 0.4792668
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2020375, 0.1438168, -0.2080763, 0.1493613, -0.3513988, 0.3518931
1: -0.1867528, 0.1502049, -0.1916938, 0.1559179, -0.3426707, 0.3418987
2: -0.1263626, 0.2461205, -0.1324517, 0.2524495, -0.3788120, 0.3785721
3: -0.1135045, 0.2968153, -0.1234874, 0.3046834, -0.4104931, 0.4125689
4: -0.1632741, 0.1942582, -0.1682969, 0.2022310, -0.3655051, 0.3625551
5: -0.1530318, 0.2291611, -0.1591532, 0.2351366, -0.3881684, 0.3883142
6: -0.1902892, 0.1826943, -0.1962010, 0.1900162, -0.3803053, 0.3788953
7: 0.5451492, 1.0985003, 0.5351154, 1.1088398, -0.5636905, 0.5633849
8: -0.1406359, 0.2667988, -0.1466027, 0.2741075, -0.4147434, 0.4134015
9: -0.1453531, 0.2593255, -0.1513267, 0.2656769, -0.4110299, 0.4106521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4740589, upper bound: 0.4740589
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4740589, upper bound: 0.4740589
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2047805, 0.1463989, -0.2091263, 0.1503158, -0.3550964, 0.3555253
1: -0.1890939, 0.1528869, -0.1925352, 0.1568974, -0.3459913, 0.3454221
2: -0.1293605, 0.2491073, -0.1334668, 0.2535309, -0.3828914, 0.3825740
3: -0.1199862, 0.3006130, -0.1248073, 0.3059977, -0.4180408, 0.4177231
4: -0.1655703, 0.1978955, -0.1691680, 0.2036187, -0.3691891, 0.3670634
5: -0.1558489, 0.2318737, -0.1602119, 0.2361802, -0.3920291, 0.3920856
6: -0.1929862, 0.1861362, -0.1972268, 0.1912714, -0.3842576, 0.3833630
7: 0.5402454, 1.1056819, 0.5334520, 1.1101909, -0.5699455, 0.5722299
8: -0.1433200, 0.2702668, -0.1476488, 0.2753505, -0.4186705, 0.4179157
9: -0.1482660, 0.2622738, -0.1523299, 0.2667730, -0.4150390, 0.4146038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4740589, upper bound: 0.4768866
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4740589, upper bound: 0.4768866
time: 1.07 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.63 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4902791, upper bound: 0.4902791
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4902791, upper bound: 0.4902791
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4902791, upper bound: 0.4952743
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4902791, upper bound: 0.4953219
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4769800, upper bound: 0.4813088
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4769800, upper bound: 0.4813088
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4769800, upper bound: 0.4844611
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4769800, upper bound: 0.4844611
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4813088, upper bound: 0.4769800
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4813088, upper bound: 0.4769800
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4813088, upper bound: 0.4792668
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4813088, upper bound: 0.4792668
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4740589, upper bound: 0.4740589
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4740589, upper bound: 0.4740589
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4740589, upper bound: 0.4768866
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4740589, upper bound: 0.4768866

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2183168, 0.1587714, -0.2183168, 0.1587714, -0.3770882, 0.3770882
1: -0.1994922, 0.1648549, -0.1994922, 0.1648549, -0.3643470, 0.3643470
2: -0.1391376, 0.2617632, -0.1391376, 0.2617632, -0.4009008, 0.4009008
3: -0.1044437, 0.3201911, -0.1044437, 0.3201911, -0.4187465, 0.4187465
4: -0.1766559, 0.2155402, -0.1766559, 0.2155402, -0.3921961, 0.3921961
5: -0.1693854, 0.2460482, -0.1693854, 0.2460482, -0.4154336, 0.4154336
6: -0.2068092, 0.2005898, -0.2068092, 0.2005898, -0.4073990, 0.4073990
7: 0.5165811, 1.0787573, 0.5165811, 1.0787573, -0.5621762, 0.5621762
8: -0.1577146, 0.2836863, -0.1577146, 0.2836863, -0.4414009, 0.4414009
9: -0.1583537, 0.2754436, -0.1583537, 0.2754436, -0.4337973, 0.4337973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4843548, upper bound: 0.4827180
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4827204, upper bound: 0.4823554
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2183168, 0.1587714, -0.2206964, 0.1619159, -0.3802327, 0.3794678
1: -0.1994922, 0.1648549, -0.2017967, 0.1670093, -0.3665015, 0.3666516
2: -0.1391376, 0.2617632, -0.1417449, 0.2646741, -0.4038117, 0.4035081
3: -0.1044437, 0.3201911, -0.1102066, 0.3223836, -0.4208960, 0.4242294
4: -0.1766559, 0.2155402, -0.1786315, 0.2187539, -0.3954098, 0.3941717
5: -0.1693854, 0.2460482, -0.1717529, 0.2491017, -0.4184871, 0.4178011
6: -0.2068092, 0.2005898, -0.2095469, 0.2035981, -0.4104072, 0.4101367
7: 0.5165811, 1.0787573, 0.5131555, 1.0853381, -0.5687571, 0.5656018
8: -0.1577146, 0.2836863, -0.1603319, 0.2867908, -0.4445055, 0.4440182
9: -0.1583537, 0.2754436, -0.1610460, 0.2780313, -0.4363850, 0.4364896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4843548, upper bound: 0.4827180
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4827204, upper bound: 0.4823554
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2206964, 0.1619159, -0.2183168, 0.1587714, -0.3794678, 0.3802327
1: -0.2017967, 0.1670093, -0.1994922, 0.1648549, -0.3666516, 0.3665015
2: -0.1417449, 0.2646741, -0.1391376, 0.2617632, -0.4035081, 0.4038117
3: -0.1102066, 0.3223836, -0.1044437, 0.3201911, -0.4242293, 0.4208960
4: -0.1786315, 0.2187539, -0.1766559, 0.2155402, -0.3941717, 0.3954098
5: -0.1717529, 0.2491017, -0.1693854, 0.2460482, -0.4178011, 0.4184871
6: -0.2095469, 0.2035981, -0.2068092, 0.2005898, -0.4101367, 0.4104072
7: 0.5131555, 1.0853381, 0.5165811, 1.0787573, -0.5656018, 0.5687571
8: -0.1603319, 0.2867908, -0.1577146, 0.2836863, -0.4440182, 0.4445055
9: -0.1610460, 0.2780313, -0.1583537, 0.2754436, -0.4364896, 0.4363850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4837218, upper bound: 0.4875582
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4823554, upper bound: 0.4874238
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2206964, 0.1619159, -0.2206964, 0.1619159, -0.3826123, 0.3826123
1: -0.2017967, 0.1670093, -0.2017967, 0.1670093, -0.3688060, 0.3688060
2: -0.1417449, 0.2646741, -0.1417449, 0.2646741, -0.4064189, 0.4064189
3: -0.1102066, 0.3223836, -0.1102066, 0.3223836, -0.4263198, 0.4263198
4: -0.1786315, 0.2187539, -0.1786315, 0.2187539, -0.3973854, 0.3973854
5: -0.1717529, 0.2491017, -0.1717529, 0.2491017, -0.4208546, 0.4208546
6: -0.2095469, 0.2035981, -0.2095469, 0.2035981, -0.4131449, 0.4131449
7: 0.5131555, 1.0853381, 0.5131555, 1.0853381, -0.5721827, 0.5721827
8: -0.1603319, 0.2867908, -0.1603319, 0.2867908, -0.4471228, 0.4471228
9: -0.1610460, 0.2780313, -0.1610460, 0.2780313, -0.4390773, 0.4390773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4837218, upper bound: 0.4876856
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4823554, upper bound: 0.4875571
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2183168, 0.1587714, -0.2020375, 0.1438168, -0.3621337, 0.3608088
1: -0.1994922, 0.1648549, -0.1867528, 0.1502049, -0.3496971, 0.3516077
2: -0.1391376, 0.2617632, -0.1263626, 0.2461205, -0.3852581, 0.3881258
3: -0.1044437, 0.3201911, -0.1135045, 0.2968153, -0.3950325, 0.4269130
4: -0.1766559, 0.2155402, -0.1632741, 0.1942582, -0.3709141, 0.3788143
5: -0.1693854, 0.2460482, -0.1530318, 0.2291611, -0.3985465, 0.3990800
6: -0.2068092, 0.2005898, -0.1902892, 0.1826943, -0.3895035, 0.3908790
7: 0.5165811, 1.0787573, 0.5451492, 1.0985003, -0.5819192, 0.5336081
8: -0.1577146, 0.2836863, -0.1406359, 0.2667988, -0.4245134, 0.4243222
9: -0.1583537, 0.2754436, -0.1453531, 0.2593255, -0.4176792, 0.4207967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4686910, upper bound: 0.4695210
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4690412
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2183168, 0.1587714, -0.2047805, 0.1463989, -0.3647158, 0.3635519
1: -0.1994922, 0.1648549, -0.1890939, 0.1528869, -0.3523790, 0.3539488
2: -0.1391376, 0.2617632, -0.1293605, 0.2491073, -0.3882449, 0.3911237
3: -0.1044437, 0.3201911, -0.1199862, 0.3006130, -0.3989896, 0.4332823
4: -0.1766559, 0.2155402, -0.1655703, 0.1978955, -0.3745514, 0.3811106
5: -0.1693854, 0.2460482, -0.1558489, 0.2318737, -0.4012591, 0.4018971
6: -0.2068092, 0.2005898, -0.1929862, 0.1861362, -0.3929454, 0.3935760
7: 0.5165811, 1.0787573, 0.5402454, 1.1056819, -0.5891008, 0.5385119
8: -0.1577146, 0.2836863, -0.1433200, 0.2702668, -0.4279815, 0.4270063
9: -0.1583537, 0.2754436, -0.1482660, 0.2622738, -0.4206275, 0.4237096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4686910, upper bound: 0.4695210
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4690412
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2206964, 0.1619159, -0.2020375, 0.1438168, -0.3645133, 0.3639534
1: -0.2017967, 0.1670093, -0.1867528, 0.1502049, -0.3520016, 0.3537621
2: -0.1417449, 0.2646741, -0.1263626, 0.2461205, -0.3878653, 0.3910367
3: -0.1102066, 0.3223836, -0.1135045, 0.2968153, -0.4005153, 0.4290625
4: -0.1786315, 0.2187539, -0.1632741, 0.1942582, -0.3728897, 0.3820280
5: -0.1717529, 0.2491017, -0.1530318, 0.2291611, -0.4009140, 0.4021335
6: -0.2095469, 0.2035981, -0.1902892, 0.1826943, -0.3922412, 0.3938872
7: 0.5131555, 1.0853381, 0.5451492, 1.0985003, -0.5853448, 0.5401889
8: -0.1603319, 0.2867908, -0.1406359, 0.2667988, -0.4271307, 0.4274268
9: -0.1610460, 0.2780313, -0.1453531, 0.2593255, -0.4203715, 0.4233844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4686438, upper bound: 0.4724096
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4722844
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2206964, 0.1619159, -0.2047805, 0.1463989, -0.3670954, 0.3666964
1: -0.2017967, 0.1670093, -0.1890939, 0.1528869, -0.3546836, 0.3561032
2: -0.1417449, 0.2646741, -0.1293605, 0.2491073, -0.3908521, 0.3940346
3: -0.1102066, 0.3223836, -0.1199862, 0.3006130, -0.4043934, 0.4353493
4: -0.1786315, 0.2187539, -0.1655703, 0.1978955, -0.3765270, 0.3843243
5: -0.1717529, 0.2491017, -0.1558489, 0.2318737, -0.4036266, 0.4049506
6: -0.2095469, 0.2035981, -0.1929862, 0.1861362, -0.3956831, 0.3965842
7: 0.5131555, 1.0853381, 0.5402454, 1.1056819, -0.5925264, 0.5450927
8: -0.1603319, 0.2867908, -0.1433200, 0.2702668, -0.4305987, 0.4301109
9: -0.1610460, 0.2780313, -0.1482660, 0.2622738, -0.4233199, 0.4262973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4686438, upper bound: 0.4724096
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4722844
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2020375, 0.1438168, -0.2183168, 0.1587714, -0.3608088, 0.3621337
1: -0.1867528, 0.1502049, -0.1994922, 0.1648549, -0.3516077, 0.3496971
2: -0.1263626, 0.2461205, -0.1391376, 0.2617632, -0.3881258, 0.3852581
3: -0.1135045, 0.2968153, -0.1044437, 0.3201911, -0.4269130, 0.3950325
4: -0.1632741, 0.1942582, -0.1766559, 0.2155402, -0.3788143, 0.3709141
5: -0.1530318, 0.2291611, -0.1693854, 0.2460482, -0.3990800, 0.3985465
6: -0.1902892, 0.1826943, -0.2068092, 0.2005898, -0.3908790, 0.3895035
7: 0.5451492, 1.0985003, 0.5165811, 1.0787573, -0.5336081, 0.5819192
8: -0.1406359, 0.2667988, -0.1577146, 0.2836863, -0.4243222, 0.4245134
9: -0.1453531, 0.2593255, -0.1583537, 0.2754436, -0.4207967, 0.4176792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4752301, upper bound: 0.4673351
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4661985
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2020375, 0.1438168, -0.2206964, 0.1619159, -0.3639534, 0.3645133
1: -0.1867528, 0.1502049, -0.2017967, 0.1670093, -0.3537621, 0.3520016
2: -0.1263626, 0.2461205, -0.1417449, 0.2646741, -0.3910367, 0.3878653
3: -0.1135045, 0.2968153, -0.1102066, 0.3223836, -0.4290624, 0.4005153
4: -0.1632741, 0.1942582, -0.1786315, 0.2187539, -0.3820280, 0.3728897
5: -0.1530318, 0.2291611, -0.1717529, 0.2491017, -0.4021335, 0.4009140
6: -0.1902892, 0.1826943, -0.2095469, 0.2035981, -0.3938872, 0.3922412
7: 0.5451492, 1.0985003, 0.5131555, 1.0853381, -0.5401889, 0.5853448
8: -0.1406359, 0.2667988, -0.1603319, 0.2867908, -0.4274268, 0.4271307
9: -0.1453531, 0.2593255, -0.1610460, 0.2780313, -0.4233844, 0.4203715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4752301, upper bound: 0.4673351
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4661985
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2047805, 0.1463989, -0.2183168, 0.1587714, -0.3635519, 0.3647158
1: -0.1890939, 0.1528869, -0.1994922, 0.1648549, -0.3539488, 0.3523790
2: -0.1293605, 0.2491073, -0.1391376, 0.2617632, -0.3911237, 0.3882449
3: -0.1199862, 0.3006130, -0.1044437, 0.3201911, -0.4332822, 0.3989896
4: -0.1655703, 0.1978955, -0.1766559, 0.2155402, -0.3811106, 0.3745514
5: -0.1558489, 0.2318737, -0.1693854, 0.2460482, -0.4018971, 0.4012591
6: -0.1929862, 0.1861362, -0.2068092, 0.2005898, -0.3935760, 0.3929454
7: 0.5402454, 1.1056819, 0.5165811, 1.0787573, -0.5385119, 0.5891008
8: -0.1433200, 0.2702668, -0.1577146, 0.2836863, -0.4270063, 0.4279815
9: -0.1482660, 0.2622738, -0.1583537, 0.2754436, -0.4237096, 0.4206275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4739346, upper bound: 0.4697777
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4686261
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2047805, 0.1463989, -0.2206964, 0.1619159, -0.3666964, 0.3670954
1: -0.1890939, 0.1528869, -0.2017967, 0.1670093, -0.3561032, 0.3546836
2: -0.1293605, 0.2491073, -0.1417449, 0.2646741, -0.3940346, 0.3908521
3: -0.1199862, 0.3006130, -0.1102066, 0.3223836, -0.4353492, 0.4043935
4: -0.1655703, 0.1978955, -0.1786315, 0.2187539, -0.3843243, 0.3765270
5: -0.1558489, 0.2318737, -0.1717529, 0.2491017, -0.4049506, 0.4036266
6: -0.1929862, 0.1861362, -0.2095469, 0.2035981, -0.3965842, 0.3956831
7: 0.5402454, 1.1056819, 0.5131555, 1.0853381, -0.5450927, 0.5925264
8: -0.1433200, 0.2702668, -0.1603319, 0.2867908, -0.4301109, 0.4305987
9: -0.1482660, 0.2622738, -0.1610460, 0.2780313, -0.4262973, 0.4233199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4739346, upper bound: 0.4698383
time: 1.20 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4687274
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2020375, 0.1438168, -0.2020375, 0.1438168, -0.3458543, 0.3458543
1: -0.1867528, 0.1502049, -0.1867528, 0.1502049, -0.3369577, 0.3369577
2: -0.1263626, 0.2461205, -0.1263626, 0.2461205, -0.3724830, 0.3724830
3: -0.1135045, 0.2968153, -0.1135045, 0.2968153, -0.4026091, 0.4026091
4: -0.1632741, 0.1942582, -0.1632741, 0.1942582, -0.3575323, 0.3575323
5: -0.1530318, 0.2291611, -0.1530318, 0.2291611, -0.3821929, 0.3821929
6: -0.1902892, 0.1826943, -0.1902892, 0.1826943, -0.3729835, 0.3729835
7: 0.5451492, 1.0985003, 0.5451492, 1.0985003, -0.5533510, 0.5533510
8: -0.1406359, 0.2667988, -0.1406359, 0.2667988, -0.4074347, 0.4074347
9: -0.1453531, 0.2593255, -0.1453531, 0.2593255, -0.4046786, 0.4046786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4662993, upper bound: 0.4629131
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4619682
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2020375, 0.1438168, -0.2047805, 0.1463989, -0.3484364, 0.3485974
1: -0.1867528, 0.1502049, -0.1890939, 0.1528869, -0.3396397, 0.3392988
2: -0.1263626, 0.2461205, -0.1293605, 0.2491073, -0.3754698, 0.3754810
3: -0.1135045, 0.2968153, -0.1199862, 0.3006130, -0.4064364, 0.4089317
4: -0.1632741, 0.1942582, -0.1655703, 0.1978955, -0.3611696, 0.3598286
5: -0.1530318, 0.2291611, -0.1558489, 0.2318737, -0.3849055, 0.3850100
6: -0.1902892, 0.1826943, -0.1929862, 0.1861362, -0.3764254, 0.3756805
7: 0.5451492, 1.0985003, 0.5402454, 1.1056819, -0.5605327, 0.5582548
8: -0.1406359, 0.2667988, -0.1433200, 0.2702668, -0.4109027, 0.4101188
9: -0.1453531, 0.2593255, -0.1482660, 0.2622738, -0.4076269, 0.4075915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4662993, upper bound: 0.4629131
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4619682
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2047805, 0.1463989, -0.2020375, 0.1438168, -0.3485974, 0.3484364
1: -0.1890939, 0.1528869, -0.1867528, 0.1502049, -0.3392988, 0.3396397
2: -0.1293605, 0.2491073, -0.1263626, 0.2461205, -0.3754810, 0.3754698
3: -0.1199862, 0.3006130, -0.1135045, 0.2968153, -0.4089317, 0.4064365
4: -0.1655703, 0.1978955, -0.1632741, 0.1942582, -0.3598286, 0.3611696
5: -0.1558489, 0.2318737, -0.1530318, 0.2291611, -0.3850100, 0.3849055
6: -0.1929862, 0.1861362, -0.1902892, 0.1826943, -0.3756805, 0.3764254
7: 0.5402454, 1.1056819, 0.5451492, 1.0985003, -0.5582548, 0.5605327
8: -0.1433200, 0.2702668, -0.1406359, 0.2667988, -0.4101188, 0.4109027
9: -0.1482660, 0.2622738, -0.1453531, 0.2593255, -0.4075915, 0.4076269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4659128, upper bound: 0.4656693
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4648096
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2047805, 0.1463989, -0.2047805, 0.1463989, -0.3511795, 0.3511795
1: -0.1890939, 0.1528869, -0.1890939, 0.1528869, -0.3419808, 0.3419808
2: -0.1293605, 0.2491073, -0.1293605, 0.2491073, -0.3784678, 0.3784678
3: -0.1199862, 0.3006130, -0.1199862, 0.3006130, -0.4126723, 0.4126722
4: -0.1655703, 0.1978955, -0.1655703, 0.1978955, -0.3634658, 0.3634658
5: -0.1558489, 0.2318737, -0.1558489, 0.2318737, -0.3877226, 0.3877226
6: -0.1929862, 0.1861362, -0.1929862, 0.1861362, -0.3791223, 0.3791223
7: 0.5402454, 1.1056819, 0.5402454, 1.1056819, -0.5654365, 0.5654365
8: -0.1433200, 0.2702668, -0.1433200, 0.2702668, -0.4135869, 0.4135869
9: -0.1482660, 0.2622738, -0.1482660, 0.2622738, -0.4105398, 0.4105398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4659128, upper bound: 0.4657014
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4648725
time: 1.07 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.57 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4843548, upper bound: 0.4827180
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4827204, upper bound: 0.4823554
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4843548, upper bound: 0.4827180
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4827204, upper bound: 0.4823554
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4837218, upper bound: 0.4875582
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4823554, upper bound: 0.4874238
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4837218, upper bound: 0.4876856
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4823554, upper bound: 0.4875571
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4686910, upper bound: 0.4695210
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4690412
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4686910, upper bound: 0.4695210
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4690412
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4686438, upper bound: 0.4724096
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4722844
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4686438, upper bound: 0.4724096
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4722844
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4752301, upper bound: 0.4673351
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4661985
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4752301, upper bound: 0.4673351
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4661985
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4739346, upper bound: 0.4697777
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4686261
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4739346, upper bound: 0.4698383
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4687274
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4662993, upper bound: 0.4629131
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4619682
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4662993, upper bound: 0.4629131
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4619682
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4659128, upper bound: 0.4656693
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4648096
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4659128, upper bound: 0.4657014
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4648725

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1957917, 0.1376383, -0.2183168, 0.1587714, -0.3545631, 0.3559552
1: -0.1817190, 0.1446229, -0.1994922, 0.1648549, -0.3465739, 0.3441150
2: -0.1198079, 0.2389458, -0.1391376, 0.2617632, -0.3815711, 0.3780834
3: -0.1027307, 0.2922405, -0.1044437, 0.3201911, -0.4165272, 0.3907057
4: -0.1581436, 0.1856094, -0.1766559, 0.2155402, -0.3736839, 0.3622653
5: -0.1468158, 0.2227110, -0.1693854, 0.2460482, -0.3928640, 0.3920964
6: -0.1843564, 0.1744584, -0.2068092, 0.2005898, -0.3849462, 0.3812676
7: 0.5513976, 1.0772233, 0.5165811, 1.0787573, -0.5273597, 0.5606422
8: -0.1343574, 0.2585509, -0.1577146, 0.2836863, -0.4180437, 0.4162655
9: -0.1385188, 0.2523187, -0.1583537, 0.2754436, -0.4139624, 0.4106723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4827204, upper bound: 0.4827204
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4827204, upper bound: 0.4827204
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1981227, 0.1397733, -0.2152528, 0.1549238, -0.3530465, 0.3550261
1: -0.1836794, 0.1469274, -0.1966910, 0.1621028, -0.3457822, 0.3436184
2: -0.1222568, 0.2413561, -0.1365072, 0.2583308, -0.3805876, 0.3778633
3: -0.1076335, 0.2956828, -0.1042053, 0.3163625, -0.4178666, 0.3940123
4: -0.1601131, 0.1886488, -0.1741373, 0.2114113, -0.3715244, 0.3627861
5: -0.1492041, 0.2250202, -0.1663157, 0.2421115, -0.3913156, 0.3913358
6: -0.1866842, 0.1772477, -0.2032555, 0.1970337, -0.3837178, 0.3805032
7: 0.5470352, 1.0812945, 0.5217073, 1.0785527, -0.5315175, 0.5595872
8: -0.1366474, 0.2613557, -0.1541841, 0.2802663, -0.4169137, 0.4155399
9: -0.1408535, 0.2547590, -0.1554712, 0.2722983, -0.4131518, 0.4102302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4827204, upper bound: 0.4827204
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4827204, upper bound: 0.4827204
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1957917, 0.1376383, -0.2206964, 0.1619159, -0.3577076, 0.3583348
1: -0.1817190, 0.1446229, -0.2017967, 0.1670093, -0.3487283, 0.3464196
2: -0.1198079, 0.2389458, -0.1417449, 0.2646741, -0.3844820, 0.3806907
3: -0.1027307, 0.2922405, -0.1102066, 0.3223836, -0.4186767, 0.3961885
4: -0.1581436, 0.1856094, -0.1786315, 0.2187539, -0.3768976, 0.3642409
5: -0.1468158, 0.2227110, -0.1717529, 0.2491017, -0.3959175, 0.3944639
6: -0.1843564, 0.1744584, -0.2095469, 0.2035981, -0.3879544, 0.3840053
7: 0.5513976, 1.0772233, 0.5131555, 1.0853381, -0.5339406, 0.5640678
8: -0.1343574, 0.2585509, -0.1603319, 0.2867908, -0.4211482, 0.4188828
9: -0.1385188, 0.2523187, -0.1610460, 0.2780313, -0.4165501, 0.4133646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4874238, upper bound: 0.4823554
time: 1.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4874238, upper bound: 0.4823554
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1981227, 0.1397733, -0.2176463, 0.1580835, -0.3562062, 0.3574196
1: -0.1836794, 0.1469274, -0.1990061, 0.1642751, -0.3479545, 0.3459335
2: -0.1222568, 0.2413561, -0.1391263, 0.2612557, -0.3835126, 0.3804824
3: -0.1076335, 0.2956828, -0.1099713, 0.3185773, -0.4200348, 0.3995247
4: -0.1601131, 0.1886488, -0.1761228, 0.2146458, -0.3747588, 0.3647715
5: -0.1492041, 0.2250202, -0.1686975, 0.2451819, -0.3943860, 0.3937177
6: -0.1866842, 0.1772477, -0.2060107, 0.2000560, -0.3867402, 0.3832584
7: 0.5470352, 1.0812945, 0.5182757, 1.0851370, -0.5381018, 0.5630188
8: -0.1366474, 0.2613557, -0.1568181, 0.2833847, -0.4200321, 0.4181738
9: -0.1408535, 0.2547590, -0.1581781, 0.2748980, -0.4157515, 0.4129371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4874238, upper bound: 0.4823554
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4874238, upper bound: 0.4823554
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1978467, 0.1396426, -0.2183168, 0.1587714, -0.3566180, 0.3579594
1: -0.1834427, 0.1464607, -0.1994922, 0.1648549, -0.3482975, 0.3459528
2: -0.1221327, 0.2412469, -0.1391376, 0.2617632, -0.3838960, 0.3803845
3: -0.1084894, 0.2939796, -0.1044437, 0.3201911, -0.4221141, 0.3924175
4: -0.1598459, 0.1883272, -0.1766559, 0.2155402, -0.3753861, 0.3649831
5: -0.1488501, 0.2247693, -0.1693854, 0.2460482, -0.3948984, 0.3941547
6: -0.1863315, 0.1770898, -0.2068092, 0.2005898, -0.3869213, 0.3838990
7: 0.5489333, 1.0838127, 0.5165811, 1.0787573, -0.5298240, 0.5672317
8: -0.1363257, 0.2612974, -0.1577146, 0.2836863, -0.4200121, 0.4190121
9: -0.1407588, 0.2545699, -0.1583537, 0.2754436, -0.4162024, 0.4129236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4823554, upper bound: 0.4874238
time: 1.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4823554, upper bound: 0.4874238
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2004130, 0.1419846, -0.2152528, 0.1549238, -0.3553368, 0.3572373
1: -0.1855939, 0.1490210, -0.1966910, 0.1621028, -0.3476967, 0.3457120
2: -0.1247960, 0.2438776, -0.1365072, 0.2583308, -0.3831268, 0.3803847
3: -0.1135312, 0.2978688, -0.1042053, 0.3163625, -0.4235828, 0.3961099
4: -0.1620215, 0.1916837, -0.1741373, 0.2114113, -0.3734329, 0.3658209
5: -0.1514867, 0.2273217, -0.1663157, 0.2421115, -0.3935982, 0.3936374
6: -0.1889065, 0.1801380, -0.2032555, 0.1970337, -0.3859401, 0.3833935
7: 0.5440536, 1.0879042, 0.5217073, 1.0785527, -0.5344992, 0.5661969
8: -0.1388672, 0.2643468, -0.1541841, 0.2802663, -0.4191335, 0.4185309
9: -0.1433101, 0.2572468, -0.1554712, 0.2722983, -0.4156084, 0.4127180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4823554, upper bound: 0.4874238
time: 1.33 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4823554, upper bound: 0.4874238
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1978467, 0.1396426, -0.2206964, 0.1619159, -0.3597625, 0.3603390
1: -0.1834427, 0.1464607, -0.2017967, 0.1670093, -0.3504519, 0.3482574
2: -0.1221327, 0.2412469, -0.1417449, 0.2646741, -0.3868068, 0.3829918
3: -0.1084894, 0.2939796, -0.1102066, 0.3223836, -0.4241996, 0.3978438
4: -0.1598459, 0.1883272, -0.1786315, 0.2187539, -0.3785998, 0.3669587
5: -0.1488501, 0.2247693, -0.1717529, 0.2491017, -0.3979518, 0.3965222
6: -0.1863315, 0.1770898, -0.2095469, 0.2035981, -0.3899295, 0.3866367
7: 0.5489333, 1.0838127, 0.5131555, 1.0853381, -0.5364048, 0.5706573
8: -0.1363257, 0.2612974, -0.1603319, 0.2867908, -0.4231166, 0.4216293
9: -0.1407588, 0.2545699, -0.1610460, 0.2780313, -0.4187901, 0.4156159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4831383, upper bound: 0.4875571
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4831383, upper bound: 0.4875571
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2004130, 0.1419846, -0.2176463, 0.1580835, -0.3584965, 0.3596309
1: -0.1855939, 0.1490210, -0.1990061, 0.1642751, -0.3498690, 0.3480271
2: -0.1247960, 0.2438776, -0.1391263, 0.2612557, -0.3860517, 0.3830038
3: -0.1135312, 0.2978688, -0.1099713, 0.3185773, -0.4256898, 0.4015589
4: -0.1620215, 0.1916837, -0.1761228, 0.2146458, -0.3766674, 0.3678064
5: -0.1514867, 0.2273217, -0.1686975, 0.2451819, -0.3966686, 0.3960192
6: -0.1889065, 0.1801380, -0.2060107, 0.2000560, -0.3889625, 0.3861487
7: 0.5440536, 1.0879042, 0.5182757, 1.0851370, -0.5410835, 0.5696285
8: -0.1388672, 0.2643468, -0.1568181, 0.2833847, -0.4222519, 0.4211649
9: -0.1433101, 0.2572468, -0.1581781, 0.2748980, -0.4182081, 0.4154249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4831383, upper bound: 0.4875571
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4831383, upper bound: 0.4875571
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1957917, 0.1376383, -0.2020375, 0.1438168, -0.3396086, 0.3396758
1: -0.1817190, 0.1446229, -0.1867528, 0.1502049, -0.3319239, 0.3313757
2: -0.1198079, 0.2389458, -0.1263626, 0.2461205, -0.3659284, 0.3653084
3: -0.1027307, 0.2922405, -0.1135045, 0.2968153, -0.3928132, 0.3988722
4: -0.1581436, 0.1856094, -0.1632741, 0.1942582, -0.3524019, 0.3488835
5: -0.1468158, 0.2227110, -0.1530318, 0.2291611, -0.3759769, 0.3757428
6: -0.1843564, 0.1744584, -0.1902892, 0.1826943, -0.3670507, 0.3647476
7: 0.5513976, 1.0772233, 0.5451492, 1.0985003, -0.5471027, 0.5320741
8: -0.1343574, 0.2585509, -0.1406359, 0.2667988, -0.4011561, 0.3991868
9: -0.1385188, 0.2523187, -0.1453531, 0.2593255, -0.3978442, 0.3976717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4690412
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4690412
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1981227, 0.1397733, -0.1989474, 0.1410962, -0.3392189, 0.3387207
1: -0.1836794, 0.1469274, -0.1843784, 0.1474147, -0.3310941, 0.3313058
2: -0.1222568, 0.2413561, -0.1237032, 0.2430500, -0.3653068, 0.3650594
3: -0.1076335, 0.2956828, -0.1133023, 0.2929325, -0.3940851, 0.4022833
4: -0.1601131, 0.1886488, -0.1607334, 0.1901614, -0.3502745, 0.3493822
5: -0.1492041, 0.2250202, -0.1499298, 0.2260992, -0.3753033, 0.3749500
6: -0.1866842, 0.1772477, -0.1872965, 0.1791069, -0.3657911, 0.3645442
7: 0.5470352, 1.0812945, 0.5499318, 1.0983133, -0.5512781, 0.5313628
8: -0.1366474, 0.2613557, -0.1374956, 0.2633512, -0.3999986, 0.3988513
9: -0.1408535, 0.2547590, -0.1426589, 0.2561544, -0.3970079, 0.3974179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4690412
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4690412
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1957917, 0.1376383, -0.2047805, 0.1463989, -0.3421906, 0.3424189
1: -0.1817190, 0.1446229, -0.1890939, 0.1528869, -0.3346059, 0.3337168
2: -0.1198079, 0.2389458, -0.1293605, 0.2491073, -0.3689151, 0.3683063
3: -0.1027307, 0.2922405, -0.1199862, 0.3006130, -0.3967703, 0.4052415
4: -0.1581436, 0.1856094, -0.1655703, 0.1978955, -0.3560391, 0.3511797
5: -0.1468158, 0.2227110, -0.1558489, 0.2318737, -0.3786895, 0.3785599
6: -0.1843564, 0.1744584, -0.1929862, 0.1861362, -0.3704926, 0.3674446
7: 0.5513976, 1.0772233, 0.5402454, 1.1056819, -0.5542843, 0.5369779
8: -0.1343574, 0.2585509, -0.1433200, 0.2702668, -0.4046242, 0.4018709
9: -0.1385188, 0.2523187, -0.1482660, 0.2622738, -0.4007926, 0.4005846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4686261, upper bound: 0.4690412
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4686261, upper bound: 0.4690412
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1981227, 0.1397733, -0.2016863, 0.1436730, -0.3417957, 0.3414596
1: -0.1836794, 0.1469274, -0.1867131, 0.1500822, -0.3337616, 0.3336405
2: -0.1222568, 0.2413561, -0.1266922, 0.2460338, -0.3682906, 0.3680484
3: -0.1076335, 0.2956828, -0.1197840, 0.2966994, -0.3980113, 0.4086756
4: -0.1601131, 0.1886488, -0.1630239, 0.1937909, -0.3539039, 0.3516726
5: -0.1492041, 0.2250202, -0.1527400, 0.2288027, -0.3780068, 0.3777602
6: -0.1866842, 0.1772477, -0.1899836, 0.1825453, -0.3692295, 0.3672313
7: 0.5470352, 1.0812945, 0.5450634, 1.1054962, -0.5584610, 0.5362312
8: -0.1366474, 0.2613557, -0.1401727, 0.2668155, -0.4034629, 0.4015284
9: -0.1408535, 0.2547590, -0.1455635, 0.2590978, -0.3999512, 0.4003225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4686261, upper bound: 0.4690412
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4686261, upper bound: 0.4690412
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1978467, 0.1396426, -0.2020375, 0.1438168, -0.3416635, 0.3416800
1: -0.1834427, 0.1464607, -0.1867528, 0.1502049, -0.3336475, 0.3332135
2: -0.1221327, 0.2412469, -0.1263626, 0.2461205, -0.3682532, 0.3676095
3: -0.1084894, 0.2939796, -0.1135045, 0.2968153, -0.3984001, 0.4005839
4: -0.1598459, 0.1883272, -0.1632741, 0.1942582, -0.3541041, 0.3516013
5: -0.1488501, 0.2247693, -0.1530318, 0.2291611, -0.3780112, 0.3778011
6: -0.1863315, 0.1770898, -0.1902892, 0.1826943, -0.3690258, 0.3673790
7: 0.5489333, 1.0838127, 0.5451492, 1.0985003, -0.5495670, 0.5386635
8: -0.1363257, 0.2612974, -0.1406359, 0.2667988, -0.4031245, 0.4019334
9: -0.1407588, 0.2545699, -0.1453531, 0.2593255, -0.4000843, 0.3999230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4722844
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4722844
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2004130, 0.1419846, -0.1989474, 0.1410962, -0.3415091, 0.3409320
1: -0.1855939, 0.1490210, -0.1843784, 0.1474147, -0.3330086, 0.3333994
2: -0.1247960, 0.2438776, -0.1237032, 0.2430500, -0.3678460, 0.3675808
3: -0.1135312, 0.2978688, -0.1133023, 0.2929325, -0.3998014, 0.4043809
4: -0.1620215, 0.1916837, -0.1607334, 0.1901614, -0.3521830, 0.3524171
5: -0.1514867, 0.2273217, -0.1499298, 0.2260992, -0.3775859, 0.3772515
6: -0.1889065, 0.1801380, -0.1872965, 0.1791069, -0.3680134, 0.3674346
7: 0.5440536, 1.0879042, 0.5499318, 1.0983133, -0.5542598, 0.5379725
8: -0.1388672, 0.2643468, -0.1374956, 0.2633512, -0.4022183, 0.4018424
9: -0.1433101, 0.2572468, -0.1426589, 0.2561544, -0.3994645, 0.3999057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4722844
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4722844
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1978467, 0.1396426, -0.2047805, 0.1463989, -0.3442456, 0.3444232
1: -0.1834427, 0.1464607, -0.1890939, 0.1528869, -0.3363295, 0.3355546
2: -0.1221327, 0.2412469, -0.1293605, 0.2491073, -0.3712400, 0.3706074
3: -0.1084894, 0.2939796, -0.1199862, 0.3006130, -0.4022733, 0.4068732
4: -0.1598459, 0.1883272, -0.1655703, 0.1978955, -0.3577413, 0.3538975
5: -0.1488501, 0.2247693, -0.1558489, 0.2318737, -0.3807238, 0.3806182
6: -0.1863315, 0.1770898, -0.1929862, 0.1861362, -0.3724676, 0.3700760
7: 0.5489333, 1.0838127, 0.5402454, 1.1056819, -0.5567486, 0.5435673
8: -0.1363257, 0.2612974, -0.1433200, 0.2702668, -0.4065926, 0.4046174
9: -0.1407588, 0.2545699, -0.1482660, 0.2622738, -0.4030327, 0.4028358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4666775, upper bound: 0.4722844
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4666775, upper bound: 0.4722844
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2004130, 0.1419846, -0.2016863, 0.1436730, -0.3440860, 0.3436708
1: -0.1855939, 0.1490210, -0.1867131, 0.1500822, -0.3356761, 0.3357341
2: -0.1247960, 0.2438776, -0.1266922, 0.2460338, -0.3708298, 0.3705698
3: -0.1135312, 0.2978688, -0.1197840, 0.2966994, -0.4036462, 0.4106922
4: -0.1620215, 0.1916837, -0.1630239, 0.1937909, -0.3558124, 0.3547075
5: -0.1514867, 0.2273217, -0.1527400, 0.2288027, -0.3802894, 0.3800617
6: -0.1889065, 0.1801380, -0.1899836, 0.1825453, -0.3714518, 0.3701217
7: 0.5440536, 1.0879042, 0.5450634, 1.1054962, -0.5614426, 0.5428408
8: -0.1388672, 0.2643468, -0.1401727, 0.2668155, -0.4056827, 0.4045195
9: -0.1433101, 0.2572468, -0.1455635, 0.2590978, -0.4024079, 0.4028103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4666775, upper bound: 0.4722844
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4666775, upper bound: 0.4722844
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1808972, 0.1257179, -0.2183168, 0.1587714, -0.3396686, 0.3440348
1: -0.1704915, 0.1301977, -0.1994922, 0.1648549, -0.3353463, 0.3296899
2: -0.1077426, 0.2246991, -0.1391376, 0.2617632, -0.3695059, 0.3638367
3: -0.1120043, 0.2710538, -0.1044437, 0.3201911, -0.4251772, 0.3692695
4: -0.1459320, 0.1661268, -0.1766559, 0.2155402, -0.3614722, 0.3427827
5: -0.1321877, 0.2071069, -0.1693854, 0.2460482, -0.3782359, 0.3764924
6: -0.1692593, 0.1582533, -0.2068092, 0.2005898, -0.3698492, 0.3650624
7: 0.5776505, 1.0971313, 0.5165811, 1.0787573, -0.5011067, 0.5805502
8: -0.1184425, 0.2428019, -0.1577146, 0.2836863, -0.4021288, 0.4005165
9: -0.1267133, 0.2370867, -0.1583537, 0.2754436, -0.4021569, 0.3954404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4661985
time: 1.10 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4661985
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1825290, 0.1272058, -0.2152528, 0.1549238, -0.3374528, 0.3424586
1: -0.1719686, 0.1321760, -0.1966910, 0.1621028, -0.3340714, 0.3288670
2: -0.1097251, 0.2265452, -0.1365072, 0.2583308, -0.3680559, 0.3630524
3: -0.1166055, 0.2735360, -0.1042053, 0.3163625, -0.4263418, 0.3715078
4: -0.1474034, 0.1682799, -0.1741373, 0.2114113, -0.3588148, 0.3424172
5: -0.1338228, 0.2090905, -0.1663157, 0.2421115, -0.3759343, 0.3754061
6: -0.1711465, 0.1601918, -0.2032555, 0.1970337, -0.3681801, 0.3634473
7: 0.5743284, 1.1006465, 0.5217073, 1.0785527, -0.5042243, 0.5789392
8: -0.1202950, 0.2449334, -0.1541841, 0.2802663, -0.4005613, 0.3991175
9: -0.1284853, 0.2390319, -0.1554712, 0.2722983, -0.4007836, 0.3945031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4661985
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4661985
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1808972, 0.1257179, -0.2206964, 0.1619159, -0.3428131, 0.3464144
1: -0.1704915, 0.1301977, -0.2017967, 0.1670093, -0.3375008, 0.3319944
2: -0.1077426, 0.2246991, -0.1417449, 0.2646741, -0.3724167, 0.3664439
3: -0.1120043, 0.2710538, -0.1102066, 0.3223836, -0.4273266, 0.3747524
4: -0.1459320, 0.1661268, -0.1786315, 0.2187539, -0.3646859, 0.3447583
5: -0.1321877, 0.2071069, -0.1717529, 0.2491017, -0.3812894, 0.3788598
6: -0.1692593, 0.1582533, -0.2095469, 0.2035981, -0.3728574, 0.3678001
7: 0.5776505, 1.0971313, 0.5131555, 1.0853381, -0.5076876, 0.5839758
8: -0.1184425, 0.2428019, -0.1603319, 0.2867908, -0.4052334, 0.4031338
9: -0.1267133, 0.2370867, -0.1610460, 0.2780313, -0.4047446, 0.3981327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4722844, upper bound: 0.4661985
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4722844, upper bound: 0.4661985
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1825290, 0.1272058, -0.2176463, 0.1580835, -0.3406125, 0.3448521
1: -0.1719686, 0.1321760, -0.1990061, 0.1642751, -0.3362437, 0.3311821
2: -0.1097251, 0.2265452, -0.1391263, 0.2612557, -0.3709808, 0.3656715
3: -0.1166055, 0.2735360, -0.1099713, 0.3185773, -0.4285098, 0.3770202
4: -0.1474034, 0.1682799, -0.1761228, 0.2146458, -0.3620493, 0.3444026
5: -0.1338228, 0.2090905, -0.1686975, 0.2451819, -0.3790047, 0.3777880
6: -0.1711465, 0.1601918, -0.2060107, 0.2000560, -0.3712025, 0.3662025
7: 0.5743284, 1.1006465, 0.5182757, 1.0851370, -0.5108086, 0.5823708
8: -0.1202950, 0.2449334, -0.1568181, 0.2833847, -0.4036797, 0.4017515
9: -0.1284853, 0.2390319, -0.1581781, 0.2748980, -0.4033833, 0.3972099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4722844, upper bound: 0.4661985
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4722844, upper bound: 0.4661985
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1829757, 0.1277481, -0.2183168, 0.1587714, -0.3417471, 0.3460650
1: -0.1724627, 0.1327230, -0.1994922, 0.1648549, -0.3373176, 0.3322152
2: -0.1104584, 0.2272168, -0.1391376, 0.2617632, -0.3722217, 0.3663544
3: -0.1184711, 0.2740996, -0.1044437, 0.3201911, -0.4316413, 0.3724196
4: -0.1478177, 0.1689606, -0.1766559, 0.2155402, -0.3633579, 0.3456165
5: -0.1343182, 0.2096441, -0.1693854, 0.2460482, -0.3803664, 0.3790295
6: -0.1716500, 0.1609033, -0.2068092, 0.2005898, -0.3722399, 0.3677125
7: 0.5735836, 1.1043050, 0.5165811, 1.0787573, -0.5051737, 0.5877240
8: -0.1208325, 0.2457149, -0.1577146, 0.2836863, -0.4045188, 0.4034296
9: -0.1291714, 0.2396820, -0.1583537, 0.2754436, -0.4046150, 0.3980356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4686261
time: 1.10 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4686261
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1847818, 0.1293581, -0.2152528, 0.1549238, -0.3397056, 0.3446108
1: -0.1740662, 0.1348209, -0.1966910, 0.1621028, -0.3361689, 0.3315119
2: -0.1125725, 0.2292156, -0.1365072, 0.2583308, -0.3709033, 0.3657227
3: -0.1230852, 0.2766161, -0.1042053, 0.3163625, -0.4327533, 0.3747264
4: -0.1494231, 0.1712954, -0.1741373, 0.2114113, -0.3608344, 0.3454327
5: -0.1360951, 0.2117969, -0.1663157, 0.2421115, -0.3782066, 0.3781126
6: -0.1736884, 0.1630063, -0.2032555, 0.1970337, -0.3707221, 0.3662618
7: 0.5702533, 1.1078570, 0.5217073, 1.0785527, -0.5082995, 0.5861497
8: -0.1228552, 0.2480431, -0.1541841, 0.2802663, -0.4031215, 0.4022273
9: -0.1310511, 0.2418063, -0.1554712, 0.2722983, -0.4033493, 0.3972774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4686261
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4686261
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1829757, 0.1277481, -0.2206964, 0.1619159, -0.3448916, 0.3484446
1: -0.1724627, 0.1327230, -0.2017967, 0.1670093, -0.3394720, 0.3345197
2: -0.1104584, 0.2272168, -0.1417449, 0.2646741, -0.3751325, 0.3689617
3: -0.1184711, 0.2740996, -0.1102066, 0.3223836, -0.4337113, 0.3778230
4: -0.1478177, 0.1689606, -0.1786315, 0.2187539, -0.3665716, 0.3475921
5: -0.1343182, 0.2096441, -0.1717529, 0.2491017, -0.3834199, 0.3813970
6: -0.1716500, 0.1609033, -0.2095469, 0.2035981, -0.3752481, 0.3704502
7: 0.5735836, 1.1043050, 0.5131555, 1.0853381, -0.5117545, 0.5911496
8: -0.1208325, 0.2457149, -0.1603319, 0.2867908, -0.4076234, 0.4060468
9: -0.1291714, 0.2396820, -0.1610460, 0.2780313, -0.4072027, 0.4007280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4692607, upper bound: 0.4687274
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4692607, upper bound: 0.4687274
time: 1.36 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1847818, 0.1293581, -0.2176463, 0.1580835, -0.3428654, 0.3470044
1: -0.1740662, 0.1348209, -0.1990061, 0.1642751, -0.3383413, 0.3338270
2: -0.1125725, 0.2292156, -0.1391263, 0.2612557, -0.3738282, 0.3683419
3: -0.1230852, 0.2766161, -0.1099713, 0.3185773, -0.4348481, 0.3801574
4: -0.1494231, 0.1712954, -0.1761228, 0.2146458, -0.3640689, 0.3474181
5: -0.1360951, 0.2117969, -0.1686975, 0.2451819, -0.3812770, 0.3804944
6: -0.1736884, 0.1630063, -0.2060107, 0.2000560, -0.3737444, 0.3690170
7: 0.5702533, 1.1078570, 0.5182757, 1.0851370, -0.5148838, 0.5895813
8: -0.1228552, 0.2480431, -0.1568181, 0.2833847, -0.4062399, 0.4048613
9: -0.1310511, 0.2418063, -0.1581781, 0.2748980, -0.4059491, 0.3999843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4692607, upper bound: 0.4687274
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4692607, upper bound: 0.4687274
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1808972, 0.1257179, -0.2020375, 0.1438168, -0.3247141, 0.3277554
1: -0.1704915, 0.1301977, -0.1867528, 0.1502049, -0.3206964, 0.3169506
2: -0.1077426, 0.2246991, -0.1263626, 0.2461205, -0.3538631, 0.3510616
3: -0.1120043, 0.2710538, -0.1135045, 0.2968153, -0.4008729, 0.3768162
4: -0.1459320, 0.1661268, -0.1632741, 0.1942582, -0.3401902, 0.3294009
5: -0.1321877, 0.2071069, -0.1530318, 0.2291611, -0.3613487, 0.3601387
6: -0.1692593, 0.1582533, -0.1902892, 0.1826943, -0.3519537, 0.3485424
7: 0.5776505, 1.0971313, 0.5451492, 1.0985003, -0.5208497, 0.5519820
8: -0.1184425, 0.2428019, -0.1406359, 0.2667988, -0.3852413, 0.3834378
9: -0.1267133, 0.2370867, -0.1453531, 0.2593255, -0.3860388, 0.3824398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4619682
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4619682
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1825290, 0.1272058, -0.1989474, 0.1410962, -0.3236252, 0.3261532
1: -0.1719686, 0.1321760, -0.1843784, 0.1474147, -0.3193833, 0.3165544
2: -0.1097251, 0.2265452, -0.1237032, 0.2430500, -0.3527751, 0.3502485
3: -0.1166055, 0.2735360, -0.1133023, 0.2929325, -0.4020042, 0.3792008
4: -0.1474034, 0.1682799, -0.1607334, 0.1901614, -0.3375649, 0.3290133
5: -0.1338228, 0.2090905, -0.1499298, 0.2260992, -0.3599220, 0.3590203
6: -0.1711465, 0.1601918, -0.1872965, 0.1791069, -0.3502534, 0.3474883
7: 0.5743284, 1.1006465, 0.5499318, 1.0983133, -0.5239849, 0.5507147
8: -0.1202950, 0.2449334, -0.1374956, 0.2633512, -0.3836462, 0.3824290
9: -0.1284853, 0.2390319, -0.1426589, 0.2561544, -0.3846397, 0.3816908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4619682
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4619682
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1808972, 0.1257179, -0.2047805, 0.1463989, -0.3272961, 0.3304985
1: -0.1704915, 0.1301977, -0.1890939, 0.1528869, -0.3233784, 0.3192917
2: -0.1077426, 0.2246991, -0.1293605, 0.2491073, -0.3568499, 0.3540596
3: -0.1120043, 0.2710538, -0.1199862, 0.3006130, -0.4047003, 0.3831388
4: -0.1459320, 0.1661268, -0.1655703, 0.1978955, -0.3438274, 0.3316971
5: -0.1321877, 0.2071069, -0.1558489, 0.2318737, -0.3640614, 0.3629559
6: -0.1692593, 0.1582533, -0.1929862, 0.1861362, -0.3553955, 0.3512394
7: 0.5776505, 1.0971313, 0.5402454, 1.1056819, -0.5280313, 0.5568858
8: -0.1184425, 0.2428019, -0.1433200, 0.2702668, -0.3887094, 0.3861219
9: -0.1267133, 0.2370867, -0.1482660, 0.2622738, -0.3889872, 0.3853527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4648096, upper bound: 0.4619682
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4648096, upper bound: 0.4619682
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1825290, 0.1272058, -0.2016863, 0.1436730, -0.3262020, 0.3288921
1: -0.1719686, 0.1321760, -0.1867131, 0.1500822, -0.3220508, 0.3188891
2: -0.1097251, 0.2265452, -0.1266922, 0.2460338, -0.3557589, 0.3532375
3: -0.1166055, 0.2735360, -0.1197840, 0.2966994, -0.4058079, 0.3855391
4: -0.1474034, 0.1682799, -0.1630239, 0.1937909, -0.3411943, 0.3313037
5: -0.1338228, 0.2090905, -0.1527400, 0.2288027, -0.3626255, 0.3618305
6: -0.1711465, 0.1601918, -0.1899836, 0.1825453, -0.3536918, 0.3501754
7: 0.5743284, 1.1006465, 0.5450634, 1.1054962, -0.5311677, 0.5555831
8: -0.1202950, 0.2449334, -0.1401727, 0.2668155, -0.3871105, 0.3851061
9: -0.1284853, 0.2390319, -0.1455635, 0.2590978, -0.3875831, 0.3845954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4648096, upper bound: 0.4619682
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4648096, upper bound: 0.4619682
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1829757, 0.1277481, -0.2020375, 0.1438168, -0.3267926, 0.3297856
1: -0.1724627, 0.1327230, -0.1867528, 0.1502049, -0.3226676, 0.3194759
2: -0.1104584, 0.2272168, -0.1263626, 0.2461205, -0.3565789, 0.3535794
3: -0.1184711, 0.2740996, -0.1135045, 0.2968153, -0.4072579, 0.3798557
4: -0.1478177, 0.1689606, -0.1632741, 0.1942582, -0.3420759, 0.3322347
5: -0.1343182, 0.2096441, -0.1530318, 0.2291611, -0.3634792, 0.3626759
6: -0.1716500, 0.1609033, -0.1902892, 0.1826943, -0.3543443, 0.3511925
7: 0.5735836, 1.1043050, 0.5451492, 1.0985003, -0.5249166, 0.5591558
8: -0.1208325, 0.2457149, -0.1406359, 0.2667988, -0.3876313, 0.3863509
9: -0.1291714, 0.2396820, -0.1453531, 0.2593255, -0.3884969, 0.3850350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4648096
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4648096
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1847818, 0.1293581, -0.1989474, 0.1410962, -0.3258780, 0.3283055
1: -0.1740662, 0.1348209, -0.1843784, 0.1474147, -0.3214809, 0.3191993
2: -0.1125725, 0.2292156, -0.1237032, 0.2430500, -0.3556225, 0.3529188
3: -0.1230852, 0.2766161, -0.1133023, 0.2929325, -0.4083580, 0.3823077
4: -0.1494231, 0.1712954, -0.1607334, 0.1901614, -0.3395846, 0.3320288
5: -0.1360951, 0.2117969, -0.1499298, 0.2260992, -0.3621943, 0.3617267
6: -0.1736884, 0.1630063, -0.1872965, 0.1791069, -0.3527953, 0.3503029
7: 0.5702533, 1.1078570, 0.5499318, 1.0983133, -0.5280601, 0.5579252
8: -0.1228552, 0.2480431, -0.1374956, 0.2633512, -0.3862064, 0.3855387
9: -0.1310511, 0.2418063, -0.1426589, 0.2561544, -0.3872055, 0.3844652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4648096
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4648096
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1829757, 0.1277481, -0.2047805, 0.1463989, -0.3293747, 0.3325287
1: -0.1724627, 0.1327230, -0.1890939, 0.1528869, -0.3253496, 0.3218170
2: -0.1104584, 0.2272168, -0.1293605, 0.2491073, -0.3595657, 0.3565773
3: -0.1184711, 0.2740996, -0.1199862, 0.3006130, -0.4109925, 0.3860909
4: -0.1478177, 0.1689606, -0.1655703, 0.1978955, -0.3457131, 0.3345309
5: -0.1343182, 0.2096441, -0.1558489, 0.2318737, -0.3661919, 0.3654930
6: -0.1716500, 0.1609033, -0.1929862, 0.1861362, -0.3577862, 0.3538895
7: 0.5735836, 1.1043050, 0.5402454, 1.1056819, -0.5320983, 0.5640596
8: -0.1208325, 0.2457149, -0.1433200, 0.2702668, -0.3910993, 0.3890349
9: -0.1291714, 0.2396820, -0.1482660, 0.2622738, -0.3914452, 0.3879479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4626084, upper bound: 0.4648725
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4626084, upper bound: 0.4648725
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1847818, 0.1293581, -0.2016863, 0.1436730, -0.3284549, 0.3310443
1: -0.1740662, 0.1348209, -0.1867131, 0.1500822, -0.3241484, 0.3215340
2: -0.1125725, 0.2292156, -0.1266922, 0.2460338, -0.3586063, 0.3559078
3: -0.1230852, 0.2766161, -0.1197840, 0.2966994, -0.4120734, 0.3885605
4: -0.1494231, 0.1712954, -0.1630239, 0.1937909, -0.3432140, 0.3343192
5: -0.1360951, 0.2117969, -0.1527400, 0.2288027, -0.3648978, 0.3645369
6: -0.1736884, 0.1630063, -0.1899836, 0.1825453, -0.3562337, 0.3529900
7: 0.5702533, 1.1078570, 0.5450634, 1.1054962, -0.5352429, 0.5627936
8: -0.1228552, 0.2480431, -0.1401727, 0.2668155, -0.3896707, 0.3882158
9: -0.1310511, 0.2418063, -0.1455635, 0.2590978, -0.3901488, 0.3873698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=35, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4626084, upper bound: 0.4648725
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4626084, upper bound: 0.4648725
time: 0.82 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.48 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4827204, upper bound: 0.4827204
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4827204, upper bound: 0.4827204
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4827204, upper bound: 0.4827204
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4827204, upper bound: 0.4827204
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4874238, upper bound: 0.4823554
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4874238, upper bound: 0.4823554
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4874238, upper bound: 0.4823554
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4874238, upper bound: 0.4823554
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4823554, upper bound: 0.4874238
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4823554, upper bound: 0.4874238
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4823554, upper bound: 0.4874238
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4823554, upper bound: 0.4874238
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4831383, upper bound: 0.4875571
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4831383, upper bound: 0.4875571
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4831383, upper bound: 0.4875571
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4831383, upper bound: 0.4875571
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4690412
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4690412
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4690412
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4690412
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4686261, upper bound: 0.4690412
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4686261, upper bound: 0.4690412
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4686261, upper bound: 0.4690412
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4686261, upper bound: 0.4690412
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4722844
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4722844
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4722844
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4661985, upper bound: 0.4722844
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4666775, upper bound: 0.4722844
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4666775, upper bound: 0.4722844
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4666775, upper bound: 0.4722844
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4666775, upper bound: 0.4722844
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4661985
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4661985
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4661985
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4661985
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4722844, upper bound: 0.4661985
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4722844, upper bound: 0.4661985
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4722844, upper bound: 0.4661985
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4722844, upper bound: 0.4661985
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4686261
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4686261
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4686261
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4690412, upper bound: 0.4686261
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4692607, upper bound: 0.4687274
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4692607, upper bound: 0.4687274
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4692607, upper bound: 0.4687274
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4692607, upper bound: 0.4687274
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4619682
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4619682
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4619682
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4619682
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4648096, upper bound: 0.4619682
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4648096, upper bound: 0.4619682
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4648096, upper bound: 0.4619682
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4648096, upper bound: 0.4619682
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4648096
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4648096
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4648096
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4619682, upper bound: 0.4648096
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4626084, upper bound: 0.4648725
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4626084, upper bound: 0.4648725
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4626084, upper bound: 0.4648725
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 7, lower bound: -0.4626084, upper bound: 0.4648725

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1957917, 0.1376383, -0.1957917, 0.1376383, -0.3334301, 0.3334301
1: -0.1817190, 0.1446229, -0.1817190, 0.1446229, -0.3263419, 0.3263419
2: -0.1198079, 0.2389458, -0.1198079, 0.2389458, -0.3587537, 0.3587537
3: -0.1027307, 0.2922405, -0.1027307, 0.2922405, -0.3884864, 0.3884864
4: -0.1581436, 0.1856094, -0.1581436, 0.1856094, -0.3437530, 0.3437530
5: -0.1468158, 0.2227110, -0.1468158, 0.2227110, -0.3695268, 0.3695268
6: -0.1843564, 0.1744584, -0.1843564, 0.1744584, -0.3588148, 0.3588148
7: 0.5513976, 1.0772233, 0.5513976, 1.0772233, -0.5258257, 0.5258257
8: -0.1343574, 0.2585509, -0.1343574, 0.2585509, -0.3929083, 0.3929083
9: -0.1385188, 0.2523187, -0.1385188, 0.2523187, -0.3908374, 0.3908374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4791408, upper bound: 0.4792314
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4819819, upper bound: 0.4804261
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1957917, 0.1376383, -0.1981227, 0.1397733, -0.3355650, 0.3357611
1: -0.1817190, 0.1446229, -0.1836794, 0.1469274, -0.3286464, 0.3283023
2: -0.1198079, 0.2389458, -0.1222568, 0.2413561, -0.3611640, 0.3612026
3: -0.1027307, 0.2922405, -0.1076335, 0.2956828, -0.3921188, 0.3936378
4: -0.1581436, 0.1856094, -0.1601131, 0.1886488, -0.3467924, 0.3457224
5: -0.1468158, 0.2227110, -0.1492041, 0.2250202, -0.3718360, 0.3719151
6: -0.1843564, 0.1744584, -0.1866842, 0.1772477, -0.3616041, 0.3611426
7: 0.5513976, 1.0772233, 0.5470352, 1.0812945, -0.5298970, 0.5301881
8: -0.1343574, 0.2585509, -0.1366474, 0.2613557, -0.3957131, 0.3951983
9: -0.1385188, 0.2523187, -0.1408535, 0.2547590, -0.3932778, 0.3931721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4791408, upper bound: 0.4792314
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4819819, upper bound: 0.4804261
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1981227, 0.1397733, -0.1957917, 0.1376383, -0.3357611, 0.3355650
1: -0.1836794, 0.1469274, -0.1817190, 0.1446229, -0.3283023, 0.3286464
2: -0.1222568, 0.2413561, -0.1198079, 0.2389458, -0.3612026, 0.3611640
3: -0.1076335, 0.2956828, -0.1027307, 0.2922405, -0.3936377, 0.3921188
4: -0.1601131, 0.1886488, -0.1581436, 0.1856094, -0.3457224, 0.3467924
5: -0.1492041, 0.2250202, -0.1468158, 0.2227110, -0.3719151, 0.3718360
6: -0.1866842, 0.1772477, -0.1843564, 0.1744584, -0.3611426, 0.3616041
7: 0.5470352, 1.0812945, 0.5513976, 1.0772233, -0.5301881, 0.5298970
8: -0.1366474, 0.2613557, -0.1343574, 0.2585509, -0.3951983, 0.3957131
9: -0.1408535, 0.2547590, -0.1385188, 0.2523187, -0.3931721, 0.3932778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4776461, upper bound: 0.4788126
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4801322, upper bound: 0.4801323
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1981227, 0.1397733, -0.1981227, 0.1397733, -0.3378960, 0.3378960
1: -0.1836794, 0.1469274, -0.1836794, 0.1469274, -0.3306068, 0.3306068
2: -0.1222568, 0.2413561, -0.1222568, 0.2413561, -0.3636129, 0.3636129
3: -0.1076335, 0.2956828, -0.1076335, 0.2956828, -0.3970697, 0.3970696
4: -0.1601131, 0.1886488, -0.1601131, 0.1886488, -0.3487618, 0.3487618
5: -0.1492041, 0.2250202, -0.1492041, 0.2250202, -0.3742243, 0.3742243
6: -0.1866842, 0.1772477, -0.1866842, 0.1772477, -0.3639318, 0.3639318
7: 0.5470352, 1.0812945, 0.5470352, 1.0812945, -0.5342593, 0.5342593
8: -0.1366474, 0.2613557, -0.1366474, 0.2613557, -0.3980032, 0.3980032
9: -0.1408535, 0.2547590, -0.1408535, 0.2547590, -0.3956125, 0.3956125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4776461, upper bound: 0.4788126
time: 1.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4801322, upper bound: 0.4801323
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1957917, 0.1376383, -0.1978467, 0.1396426, -0.3354343, 0.3354850
1: -0.1817190, 0.1446229, -0.1834427, 0.1464607, -0.3281797, 0.3280655
2: -0.1198079, 0.2389458, -0.1221327, 0.2412469, -0.3610548, 0.3610785
3: -0.1027307, 0.2922405, -0.1084894, 0.2939796, -0.3901981, 0.3940733
4: -0.1581436, 0.1856094, -0.1598459, 0.1883272, -0.3464708, 0.3454552
5: -0.1468158, 0.2227110, -0.1488501, 0.2247693, -0.3715851, 0.3715611
6: -0.1843564, 0.1744584, -0.1863315, 0.1770898, -0.3614462, 0.3607899
7: 0.5513976, 1.0772233, 0.5489333, 1.0838127, -0.5324152, 0.5282900
8: -0.1343574, 0.2585509, -0.1363257, 0.2612974, -0.3956548, 0.3948766
9: -0.1385188, 0.2523187, -0.1407588, 0.2545699, -0.3930886, 0.3930774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4836484, upper bound: 0.4788719
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4866729, upper bound: 0.4801987
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1957917, 0.1376383, -0.2004130, 0.1419846, -0.3377762, 0.3380513
1: -0.1817190, 0.1446229, -0.1855939, 0.1490210, -0.3307400, 0.3302168
2: -0.1198079, 0.2389458, -0.1247960, 0.2438776, -0.3636854, 0.3637418
3: -0.1027307, 0.2922405, -0.1135312, 0.2978688, -0.3942082, 0.3993540
4: -0.1581436, 0.1856094, -0.1620215, 0.1916837, -0.3498273, 0.3476309
5: -0.1468158, 0.2227110, -0.1514867, 0.2273217, -0.3741375, 0.3741977
6: -0.1843564, 0.1744584, -0.1889065, 0.1801380, -0.3644944, 0.3633649
7: 0.5513976, 1.0772233, 0.5440536, 1.0879042, -0.5365067, 0.5331697
8: -0.1343574, 0.2585509, -0.1388672, 0.2643468, -0.3987042, 0.3974181
9: -0.1385188, 0.2523187, -0.1433101, 0.2572468, -0.3957656, 0.3956288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4836484, upper bound: 0.4788719
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4866729, upper bound: 0.4801987
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1981227, 0.1397733, -0.1978467, 0.1396426, -0.3377653, 0.3376200
1: -0.1836794, 0.1469274, -0.1834427, 0.1464607, -0.3301401, 0.3303701
2: -0.1222568, 0.2413561, -0.1221327, 0.2412469, -0.3635037, 0.3634889
3: -0.1076335, 0.2956828, -0.1084894, 0.2939796, -0.3953496, 0.3977057
4: -0.1601131, 0.1886488, -0.1598459, 0.1883272, -0.3484402, 0.3484946
5: -0.1492041, 0.2250202, -0.1488501, 0.2247693, -0.3739734, 0.3738703
6: -0.1866842, 0.1772477, -0.1863315, 0.1770898, -0.3637740, 0.3635791
7: 0.5470352, 1.0812945, 0.5489333, 1.0838127, -0.5367775, 0.5323613
8: -0.1366474, 0.2613557, -0.1363257, 0.2612974, -0.3979449, 0.3976815
9: -0.1408535, 0.2547590, -0.1407588, 0.2545699, -0.3954234, 0.3955178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4818949, upper bound: 0.4783912
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4846592, upper bound: 0.4798247
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1981227, 0.1397733, -0.2004130, 0.1419846, -0.3401073, 0.3401863
1: -0.1836794, 0.1469274, -0.1855939, 0.1490210, -0.3327005, 0.3325213
2: -0.1222568, 0.2413561, -0.1247960, 0.2438776, -0.3661344, 0.3661521
3: -0.1076335, 0.2956828, -0.1135312, 0.2978688, -0.3991673, 0.4027835
4: -0.1601131, 0.1886488, -0.1620215, 0.1916837, -0.3517967, 0.3506703
5: -0.1492041, 0.2250202, -0.1514867, 0.2273217, -0.3765258, 0.3765069
6: -0.1866842, 0.1772477, -0.1889065, 0.1801380, -0.3668222, 0.3661541
7: 0.5470352, 1.0812945, 0.5440536, 1.0879042, -0.5408690, 0.5372410
8: -0.1366474, 0.2613557, -0.1388672, 0.2643468, -0.4009942, 0.4002229
9: -0.1408535, 0.2547590, -0.1433101, 0.2572468, -0.3981003, 0.3980691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4818949, upper bound: 0.4783912
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4846592, upper bound: 0.4798247
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1978467, 0.1396426, -0.1957917, 0.1376383, -0.3354850, 0.3354343
1: -0.1834427, 0.1464607, -0.1817190, 0.1446229, -0.3280655, 0.3281797
2: -0.1221327, 0.2412469, -0.1198079, 0.2389458, -0.3610785, 0.3610548
3: -0.1084894, 0.2939796, -0.1027307, 0.2922405, -0.3940733, 0.3901982
4: -0.1598459, 0.1883272, -0.1581436, 0.1856094, -0.3454552, 0.3464708
5: -0.1488501, 0.2247693, -0.1468158, 0.2227110, -0.3715611, 0.3715851
6: -0.1863315, 0.1770898, -0.1843564, 0.1744584, -0.3607899, 0.3614462
7: 0.5489333, 1.0838127, 0.5513976, 1.0772233, -0.5282900, 0.5324152
8: -0.1363257, 0.2612974, -0.1343574, 0.2585509, -0.3948766, 0.3956548
9: -0.1407588, 0.2545699, -0.1385188, 0.2523187, -0.3930774, 0.3930886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4788160, upper bound: 0.4839497
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4812841, upper bound: 0.4847832
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1978467, 0.1396426, -0.1981227, 0.1397733, -0.3376200, 0.3377653
1: -0.1834427, 0.1464607, -0.1836794, 0.1469274, -0.3303701, 0.3301401
2: -0.1221327, 0.2412469, -0.1222568, 0.2413561, -0.3634889, 0.3635037
3: -0.1084894, 0.2939796, -0.1076335, 0.2956828, -0.3977058, 0.3953495
4: -0.1598459, 0.1883272, -0.1601131, 0.1886488, -0.3484946, 0.3484402
5: -0.1488501, 0.2247693, -0.1492041, 0.2250202, -0.3738703, 0.3739734
6: -0.1863315, 0.1770898, -0.1866842, 0.1772477, -0.3635791, 0.3637740
7: 0.5489333, 1.0838127, 0.5470352, 1.0812945, -0.5323613, 0.5367775
8: -0.1363257, 0.2612974, -0.1366474, 0.2613557, -0.3976815, 0.3979449
9: -0.1407588, 0.2545699, -0.1408535, 0.2547590, -0.3955178, 0.3954234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4788160, upper bound: 0.4839497
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4812841, upper bound: 0.4847832
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2004130, 0.1419846, -0.1957917, 0.1376383, -0.3380513, 0.3377762
1: -0.1855939, 0.1490210, -0.1817190, 0.1446229, -0.3302168, 0.3307400
2: -0.1247960, 0.2438776, -0.1198079, 0.2389458, -0.3637418, 0.3636854
3: -0.1135312, 0.2978688, -0.1027307, 0.2922405, -0.3993540, 0.3942081
4: -0.1620215, 0.1916837, -0.1581436, 0.1856094, -0.3476309, 0.3498273
5: -0.1514867, 0.2273217, -0.1468158, 0.2227110, -0.3741977, 0.3741375
6: -0.1889065, 0.1801380, -0.1843564, 0.1744584, -0.3633649, 0.3644944
7: 0.5440536, 1.0879042, 0.5513976, 1.0772233, -0.5331697, 0.5365067
8: -0.1388672, 0.2643468, -0.1343574, 0.2585509, -0.3974181, 0.3987042
9: -0.1433101, 0.2572468, -0.1385188, 0.2523187, -0.3956288, 0.3957656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4774562, upper bound: 0.4839114
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4798247, upper bound: 0.4846592
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2004130, 0.1419846, -0.1981227, 0.1397733, -0.3401863, 0.3401073
1: -0.1855939, 0.1490210, -0.1836794, 0.1469274, -0.3325213, 0.3327005
2: -0.1247960, 0.2438776, -0.1222568, 0.2413561, -0.3661521, 0.3661344
3: -0.1135312, 0.2978688, -0.1076335, 0.2956828, -0.4027836, 0.3991672
4: -0.1620215, 0.1916837, -0.1601131, 0.1886488, -0.3506703, 0.3517967
5: -0.1514867, 0.2273217, -0.1492041, 0.2250202, -0.3765069, 0.3765258
6: -0.1889065, 0.1801380, -0.1866842, 0.1772477, -0.3661541, 0.3668222
7: 0.5440536, 1.0879042, 0.5470352, 1.0812945, -0.5372410, 0.5408690
8: -0.1388672, 0.2643468, -0.1366474, 0.2613557, -0.4002229, 0.4009942
9: -0.1433101, 0.2572468, -0.1408535, 0.2547590, -0.3980691, 0.3981003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4774562, upper bound: 0.4839114
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4798247, upper bound: 0.4846592
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1978467, 0.1396426, -0.1978467, 0.1396426, -0.3374892, 0.3374892
1: -0.1834427, 0.1464607, -0.1834427, 0.1464607, -0.3299033, 0.3299033
2: -0.1221327, 0.2412469, -0.1221327, 0.2412469, -0.3633796, 0.3633796
3: -0.1084894, 0.2939796, -0.1084894, 0.2939796, -0.3957238, 0.3957237
4: -0.1598459, 0.1883272, -0.1598459, 0.1883272, -0.3481731, 0.3481731
5: -0.1488501, 0.2247693, -0.1488501, 0.2247693, -0.3736194, 0.3736194
6: -0.1863315, 0.1770898, -0.1863315, 0.1770898, -0.3634213, 0.3634213
7: 0.5489333, 1.0838127, 0.5489333, 1.0838127, -0.5348794, 0.5348794
8: -0.1363257, 0.2612974, -0.1363257, 0.2612974, -0.3976232, 0.3976232
9: -0.1407588, 0.2545699, -0.1407588, 0.2545699, -0.3953287, 0.3953287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4793078, upper bound: 0.4840348
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4817020, upper bound: 0.4848903
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1978467, 0.1396426, -0.2004130, 0.1419846, -0.3398312, 0.3400556
1: -0.1834427, 0.1464607, -0.1855939, 0.1490210, -0.3324637, 0.3320546
2: -0.1221327, 0.2412469, -0.1247960, 0.2438776, -0.3660103, 0.3660429
3: -0.1084894, 0.2939796, -0.1135312, 0.2978688, -0.3997334, 0.4010065
4: -0.1598459, 0.1883272, -0.1620215, 0.1916837, -0.3515295, 0.3503487
5: -0.1488501, 0.2247693, -0.1514867, 0.2273217, -0.3761719, 0.3762560
6: -0.1863315, 0.1770898, -0.1889065, 0.1801380, -0.3664695, 0.3659963
7: 0.5489333, 1.0838127, 0.5440536, 1.0879042, -0.5389709, 0.5397592
8: -0.1363257, 0.2612974, -0.1388672, 0.2643468, -0.4006725, 0.4001646
9: -0.1407588, 0.2545699, -0.1433101, 0.2572468, -0.3980056, 0.3978800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4793078, upper bound: 0.4840348
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4817020, upper bound: 0.4848903
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2004130, 0.1419846, -0.1978467, 0.1396426, -0.3400556, 0.3398312
1: -0.1855939, 0.1490210, -0.1834427, 0.1464607, -0.3320546, 0.3324637
2: -0.1247960, 0.2438776, -0.1221327, 0.2412469, -0.3660429, 0.3660103
3: -0.1135312, 0.2978688, -0.1084894, 0.2939796, -0.4010064, 0.3997335
4: -0.1620215, 0.1916837, -0.1598459, 0.1883272, -0.3503487, 0.3515295
5: -0.1514867, 0.2273217, -0.1488501, 0.2247693, -0.3762560, 0.3761719
6: -0.1889065, 0.1801380, -0.1863315, 0.1770898, -0.3659963, 0.3664695
7: 0.5440536, 1.0879042, 0.5489333, 1.0838127, -0.5397592, 0.5389709
8: -0.1388672, 0.2643468, -0.1363257, 0.2612974, -0.4001646, 0.4006725
9: -0.1433101, 0.2572468, -0.1407588, 0.2545699, -0.3978800, 0.3980056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4781874, upper bound: 0.4840080
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4805440, upper bound: 0.4848034
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2004130, 0.1419846, -0.2004130, 0.1419846, -0.3423975, 0.3423975
1: -0.1855939, 0.1490210, -0.1855939, 0.1490210, -0.3346150, 0.3346150
2: -0.1247960, 0.2438776, -0.1247960, 0.2438776, -0.3686736, 0.3686736
3: -0.1135312, 0.2978688, -0.1135312, 0.2978688, -0.4048157, 0.4048157
4: -0.1620215, 0.1916837, -0.1620215, 0.1916837, -0.3537052, 0.3537052
5: -0.1514867, 0.2273217, -0.1514867, 0.2273217, -0.3788084, 0.3788084
6: -0.1889065, 0.1801380, -0.1889065, 0.1801380, -0.3690445, 0.3690445
7: 0.5440536, 1.0879042, 0.5440536, 1.0879042, -0.5438507, 0.5438507
8: -0.1388672, 0.2643468, -0.1388672, 0.2643468, -0.4032139, 0.4032139
9: -0.1433101, 0.2572468, -0.1433101, 0.2572468, -0.4005569, 0.4005569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4781874, upper bound: 0.4840080
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4805440, upper bound: 0.4848034
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1957917, 0.1376383, -0.1808972, 0.1257179, -0.3215096, 0.3185356
1: -0.1817190, 0.1446229, -0.1704915, 0.1301977, -0.3119167, 0.3151144
2: -0.1198079, 0.2389458, -0.1077426, 0.2246991, -0.3445069, 0.3466884
3: -0.1027307, 0.2922405, -0.1120043, 0.2710538, -0.3670502, 0.3971364
4: -0.1581436, 0.1856094, -0.1459320, 0.1661268, -0.3242704, 0.3315413
5: -0.1468158, 0.2227110, -0.1321877, 0.2071069, -0.3539227, 0.3548986
6: -0.1843564, 0.1744584, -0.1692593, 0.1582533, -0.3426096, 0.3437178
7: 0.5513976, 1.0772233, 0.5776505, 1.0971313, -0.5457337, 0.4995728
8: -0.1343574, 0.2585509, -0.1184425, 0.2428019, -0.3771593, 0.3769934
9: -0.1385188, 0.2523187, -0.1267133, 0.2370867, -0.3756055, 0.3790320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4626142, upper bound: 0.4662597
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4664864, upper bound: 0.4672148
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1957917, 0.1376383, -0.1825290, 0.1272058, -0.3229975, 0.3201674
1: -0.1817190, 0.1446229, -0.1719686, 0.1321760, -0.3138950, 0.3165915
2: -0.1198079, 0.2389458, -0.1097251, 0.2265452, -0.3463531, 0.3486709
3: -0.1027307, 0.2922405, -0.1166055, 0.2735360, -0.3695592, 0.4021130
4: -0.1581436, 0.1856094, -0.1474034, 0.1682799, -0.3264235, 0.3330128
5: -0.1468158, 0.2227110, -0.1338228, 0.2090905, -0.3559063, 0.3565338
6: -0.1843564, 0.1744584, -0.1711465, 0.1601918, -0.3445482, 0.3456049
7: 0.5513976, 1.0772233, 0.5743284, 1.1006465, -0.5492489, 0.5028949
8: -0.1343574, 0.2585509, -0.1202950, 0.2449334, -0.3792908, 0.3788459
9: -0.1385188, 0.2523187, -0.1284853, 0.2390319, -0.3775506, 0.3808040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4626142, upper bound: 0.4662597
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4664864, upper bound: 0.4672148
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1981227, 0.1397733, -0.1808972, 0.1257179, -0.3238406, 0.3206705
1: -0.1836794, 0.1469274, -0.1704915, 0.1301977, -0.3138772, 0.3174189
2: -0.1222568, 0.2413561, -0.1077426, 0.2246991, -0.3469559, 0.3490988
3: -0.1076335, 0.2956828, -0.1120043, 0.2710538, -0.3722016, 0.4007688
4: -0.1601131, 0.1886488, -0.1459320, 0.1661268, -0.3262398, 0.3345807
5: -0.1492041, 0.2250202, -0.1321877, 0.2071069, -0.3563110, 0.3572078
6: -0.1866842, 0.1772477, -0.1692593, 0.1582533, -0.3449374, 0.3465070
7: 0.5470352, 1.0812945, 0.5776505, 1.0971313, -0.5500960, 0.5036440
8: -0.1366474, 0.2613557, -0.1184425, 0.2428019, -0.3794493, 0.3797982
9: -0.1408535, 0.2547590, -0.1267133, 0.2370867, -0.3779402, 0.3814723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4601998, upper bound: 0.4654807
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4638896, upper bound: 0.4667387
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1981227, 0.1397733, -0.1825290, 0.1272058, -0.3253285, 0.3223023
1: -0.1836794, 0.1469274, -0.1719686, 0.1321760, -0.3158555, 0.3188960
2: -0.1222568, 0.2413561, -0.1097251, 0.2265452, -0.3488021, 0.3510812
3: -0.1076335, 0.2956828, -0.1166055, 0.2735360, -0.3745650, 0.4055593
4: -0.1601131, 0.1886488, -0.1474034, 0.1682799, -0.3283929, 0.3360522
5: -0.1492041, 0.2250202, -0.1338228, 0.2090905, -0.3582946, 0.3588430
6: -0.1866842, 0.1772477, -0.1711465, 0.1601918, -0.3468760, 0.3483941
7: 0.5470352, 1.0812945, 0.5743284, 1.1006465, -0.5536113, 0.5069661
8: -0.1366474, 0.2613557, -0.1202950, 0.2449334, -0.3815808, 0.3816507
9: -0.1408535, 0.2547590, -0.1284853, 0.2390319, -0.3798854, 0.3832443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4601998, upper bound: 0.4654807
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4638896, upper bound: 0.4667387
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1957917, 0.1376383, -0.1829757, 0.1277481, -0.3235398, 0.3206141
1: -0.1817190, 0.1446229, -0.1724627, 0.1327230, -0.3144420, 0.3170856
2: -0.1198079, 0.2389458, -0.1104584, 0.2272168, -0.3470247, 0.3494042
3: -0.1027307, 0.2922405, -0.1184711, 0.2740996, -0.3702002, 0.4036006
4: -0.1581436, 0.1856094, -0.1478177, 0.1689606, -0.3271042, 0.3334270
5: -0.1468158, 0.2227110, -0.1343182, 0.2096441, -0.3564599, 0.3570291
6: -0.1843564, 0.1744584, -0.1716500, 0.1609033, -0.3452597, 0.3461084
7: 0.5513976, 1.0772233, 0.5735836, 1.1043050, -0.5529075, 0.5036397
8: -0.1343574, 0.2585509, -0.1208325, 0.2457149, -0.3800723, 0.3793834
9: -0.1385188, 0.2523187, -0.1291714, 0.2396820, -0.3782007, 0.3814901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4645760, upper bound: 0.4662597
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4686553, upper bound: 0.4672148
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1957917, 0.1376383, -0.1847818, 0.1293581, -0.3251498, 0.3224202
1: -0.1817190, 0.1446229, -0.1740662, 0.1348209, -0.3165399, 0.3186890
2: -0.1198079, 0.2389458, -0.1125725, 0.2292156, -0.3490235, 0.3515183
3: -0.1027307, 0.2922405, -0.1230852, 0.2766161, -0.3727831, 0.4085245
4: -0.1581436, 0.1856094, -0.1494231, 0.1712954, -0.3294390, 0.3350325
5: -0.1468158, 0.2227110, -0.1360951, 0.2117969, -0.3586127, 0.3588061
6: -0.1843564, 0.1744584, -0.1736884, 0.1630063, -0.3473627, 0.3481468
7: 0.5513976, 1.0772233, 0.5702533, 1.1078570, -0.5564594, 0.5069700
8: -0.1343574, 0.2585509, -0.1228552, 0.2480431, -0.3824005, 0.3814061
9: -0.1385188, 0.2523187, -0.1310511, 0.2418063, -0.3803250, 0.3833697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4645760, upper bound: 0.4662597
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4686553, upper bound: 0.4672148
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1981227, 0.1397733, -0.1829757, 0.1277481, -0.3258708, 0.3227490
1: -0.1836794, 0.1469274, -0.1724627, 0.1327230, -0.3164025, 0.3193901
2: -0.1222568, 0.2413561, -0.1104584, 0.2272168, -0.3494736, 0.3518146
3: -0.1076335, 0.2956828, -0.1184711, 0.2740996, -0.3753517, 0.4072330
4: -0.1601131, 0.1886488, -0.1478177, 0.1689606, -0.3290737, 0.3364664
5: -0.1492041, 0.2250202, -0.1343182, 0.2096441, -0.3588482, 0.3593383
6: -0.1866842, 0.1772477, -0.1716500, 0.1609033, -0.3475875, 0.3488977
7: 0.5470352, 1.0812945, 0.5735836, 1.1043050, -0.5572698, 0.5077109
8: -0.1366474, 0.2613557, -0.1208325, 0.2457149, -0.3823624, 0.3821883
9: -0.1408535, 0.2547590, -0.1291714, 0.2396820, -0.3805355, 0.3839304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4626839, upper bound: 0.4654708
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4662509, upper bound: 0.4667387
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1981227, 0.1397733, -0.1847818, 0.1293581, -0.3274808, 0.3245552
1: -0.1836794, 0.1469274, -0.1740662, 0.1348209, -0.3185004, 0.3209936
2: -0.1222568, 0.2413561, -0.1125725, 0.2292156, -0.3514724, 0.3539286
3: -0.1076335, 0.2956828, -0.1230852, 0.2766161, -0.3777838, 0.4119829
4: -0.1601131, 0.1886488, -0.1494231, 0.1712954, -0.3314084, 0.3380719
5: -0.1492041, 0.2250202, -0.1360951, 0.2117969, -0.3610010, 0.3611153
6: -0.1866842, 0.1772477, -0.1736884, 0.1630063, -0.3496905, 0.3509361
7: 0.5470352, 1.0812945, 0.5702533, 1.1078570, -0.5608218, 0.5110413
8: -0.1366474, 0.2613557, -0.1228552, 0.2480431, -0.3846906, 0.3842109
9: -0.1408535, 0.2547590, -0.1310511, 0.2418063, -0.3826598, 0.3858101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4626839, upper bound: 0.4654708
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4662509, upper bound: 0.4667387
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1978467, 0.1396426, -0.1808972, 0.1257179, -0.3235646, 0.3205398
1: -0.1834427, 0.1464607, -0.1704915, 0.1301977, -0.3136404, 0.3169521
2: -0.1221327, 0.2412469, -0.1077426, 0.2246991, -0.3468318, 0.3489895
3: -0.1084894, 0.2939796, -0.1120043, 0.2710538, -0.3726372, 0.3988482
4: -0.1598459, 0.1883272, -0.1459320, 0.1661268, -0.3259726, 0.3342592
5: -0.1488501, 0.2247693, -0.1321877, 0.2071069, -0.3559571, 0.3569570
6: -0.1863315, 0.1770898, -0.1692593, 0.1582533, -0.3445847, 0.3463492
7: 0.5489333, 1.0838127, 0.5776505, 1.0971313, -0.5481980, 0.5061622
8: -0.1363257, 0.2612974, -0.1184425, 0.2428019, -0.3791277, 0.3797399
9: -0.1407588, 0.2545699, -0.1267133, 0.2370867, -0.3778455, 0.3812832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4626142, upper bound: 0.4693404
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4664144, upper bound: 0.4699818
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1978467, 0.1396426, -0.1825290, 0.1272058, -0.3250525, 0.3221716
1: -0.1834427, 0.1464607, -0.1719686, 0.1321760, -0.3156187, 0.3184293
2: -0.1221327, 0.2412469, -0.1097251, 0.2265452, -0.3486780, 0.3509720
3: -0.1084894, 0.2939796, -0.1166055, 0.2735360, -0.3751462, 0.4038247
4: -0.1598459, 0.1883272, -0.1474034, 0.1682799, -0.3281257, 0.3357306
5: -0.1488501, 0.2247693, -0.1338228, 0.2090905, -0.3579406, 0.3585921
6: -0.1863315, 0.1770898, -0.1711465, 0.1601918, -0.3465232, 0.3482363
7: 0.5489333, 1.0838127, 0.5743284, 1.1006465, -0.5517132, 0.5094843
8: -0.1363257, 0.2612974, -0.1202950, 0.2449334, -0.3812591, 0.3815925
9: -0.1407588, 0.2545699, -0.1284853, 0.2390319, -0.3797907, 0.3830552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4626142, upper bound: 0.4693404
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4664144, upper bound: 0.4699818
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2004130, 0.1419846, -0.1808972, 0.1257179, -0.3261309, 0.3228818
1: -0.1855939, 0.1490210, -0.1704915, 0.1301977, -0.3157917, 0.3195125
2: -0.1247960, 0.2438776, -0.1077426, 0.2246991, -0.3494951, 0.3516202
3: -0.1135312, 0.2978688, -0.1120043, 0.2710538, -0.3779178, 0.4028582
4: -0.1620215, 0.1916837, -0.1459320, 0.1661268, -0.3281483, 0.3376156
5: -0.1514867, 0.2273217, -0.1321877, 0.2071069, -0.3585936, 0.3595094
6: -0.1889065, 0.1801380, -0.1692593, 0.1582533, -0.3471597, 0.3493974
7: 0.5440536, 1.0879042, 0.5776505, 1.0971313, -0.5530777, 0.5102537
8: -0.1388672, 0.2643468, -0.1184425, 0.2428019, -0.3816691, 0.3827893
9: -0.1433101, 0.2572468, -0.1267133, 0.2370867, -0.3803968, 0.3839601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4601996, upper bound: 0.4692252
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4638896, upper bound: 0.4698704
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2004130, 0.1419846, -0.1825290, 0.1272058, -0.3276188, 0.3245136
1: -0.1855939, 0.1490210, -0.1719686, 0.1321760, -0.3177700, 0.3209896
2: -0.1247960, 0.2438776, -0.1097251, 0.2265452, -0.3513412, 0.3536026
3: -0.1135312, 0.2978688, -0.1166055, 0.2735360, -0.3802789, 0.4076569
4: -0.1620215, 0.1916837, -0.1474034, 0.1682799, -0.3303014, 0.3390871
5: -0.1514867, 0.2273217, -0.1338228, 0.2090905, -0.3605772, 0.3611445
6: -0.1889065, 0.1801380, -0.1711465, 0.1601918, -0.3490983, 0.3512845
7: 0.5440536, 1.0879042, 0.5743284, 1.1006465, -0.5565929, 0.5135758
8: -0.1388672, 0.2643468, -0.1202950, 0.2449334, -0.3838006, 0.3846418
9: -0.1433101, 0.2572468, -0.1284853, 0.2390319, -0.3823420, 0.3857321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4601996, upper bound: 0.4692252
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4638896, upper bound: 0.4698704
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1978467, 0.1396426, -0.1829757, 0.1277481, -0.3255948, 0.3226184
1: -0.1834427, 0.1464607, -0.1724627, 0.1327230, -0.3161657, 0.3189234
2: -0.1221327, 0.2412469, -0.1104584, 0.2272168, -0.3493495, 0.3517053
3: -0.1084894, 0.2939796, -0.1184711, 0.2740996, -0.3757029, 0.4052353
4: -0.1598459, 0.1883272, -0.1478177, 0.1689606, -0.3288065, 0.3361449
5: -0.1488501, 0.2247693, -0.1343182, 0.2096441, -0.3584942, 0.3590874
6: -0.1863315, 0.1770898, -0.1716500, 0.1609033, -0.3472348, 0.3487399
7: 0.5489333, 1.0838127, 0.5735836, 1.1043050, -0.5553718, 0.5102291
8: -0.1363257, 0.2612974, -0.1208325, 0.2457149, -0.3820407, 0.3821300
9: -0.1407588, 0.2545699, -0.1291714, 0.2396820, -0.3804408, 0.3837413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4628442, upper bound: 0.4693404
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4665851, upper bound: 0.4699818
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1978467, 0.1396426, -0.1847818, 0.1293581, -0.3272047, 0.3244244
1: -0.1834427, 0.1464607, -0.1740662, 0.1348209, -0.3182636, 0.3205268
2: -0.1221327, 0.2412469, -0.1125725, 0.2292156, -0.3513483, 0.3538194
3: -0.1084894, 0.2939796, -0.1230852, 0.2766161, -0.3782906, 0.4101649
4: -0.1598459, 0.1883272, -0.1494231, 0.1712954, -0.3311412, 0.3377503
5: -0.1488501, 0.2247693, -0.1360951, 0.2117969, -0.3606470, 0.3608644
6: -0.1863315, 0.1770898, -0.1736884, 0.1630063, -0.3493378, 0.3507783
7: 0.5489333, 1.0838127, 0.5702533, 1.1078570, -0.5589237, 0.5135595
8: -0.1363257, 0.2612974, -0.1228552, 0.2480431, -0.3843689, 0.3841526
9: -0.1407588, 0.2545699, -0.1310511, 0.2418063, -0.3825651, 0.3856210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4628442, upper bound: 0.4693404
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4665851, upper bound: 0.4699818
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2004130, 0.1419846, -0.1829757, 0.1277481, -0.3281611, 0.3249603
1: -0.1855939, 0.1490210, -0.1724627, 0.1327230, -0.3183170, 0.3214837
2: -0.1247960, 0.2438776, -0.1104584, 0.2272168, -0.3520128, 0.3543360
3: -0.1135312, 0.2978688, -0.1184711, 0.2740996, -0.3809857, 0.4092451
4: -0.1620215, 0.1916837, -0.1478177, 0.1689606, -0.3309821, 0.3395013
5: -0.1514867, 0.2273217, -0.1343182, 0.2096441, -0.3611308, 0.3616399
6: -0.1889065, 0.1801380, -0.1716500, 0.1609033, -0.3498098, 0.3517880
7: 0.5440536, 1.0879042, 0.5735836, 1.1043050, -0.5602515, 0.5143206
8: -0.1388672, 0.2643468, -0.1208325, 0.2457149, -0.3845821, 0.3851793
9: -0.1433101, 0.2572468, -0.1291714, 0.2396820, -0.3829921, 0.3864182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4607191, upper bound: 0.4692252
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4643910, upper bound: 0.4698704
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2004130, 0.1419846, -0.1847818, 0.1293581, -0.3297710, 0.3267664
1: -0.1855939, 0.1490210, -0.1740662, 0.1348209, -0.3204149, 0.3230872
2: -0.1247960, 0.2438776, -0.1125725, 0.2292156, -0.3540116, 0.3564500
3: -0.1135312, 0.2978688, -0.1230852, 0.2766161, -0.3834143, 0.4140024
4: -0.1620215, 0.1916837, -0.1494231, 0.1712954, -0.3333169, 0.3411068
5: -0.1514867, 0.2273217, -0.1360951, 0.2117969, -0.3632836, 0.3634169
6: -0.1889065, 0.1801380, -0.1736884, 0.1630063, -0.3519128, 0.3538264
7: 0.5440536, 1.0879042, 0.5702533, 1.1078570, -0.5638034, 0.5176510
8: -0.1388672, 0.2643468, -0.1228552, 0.2480431, -0.3869103, 0.3872020
9: -0.1433101, 0.2572468, -0.1310511, 0.2418063, -0.3851164, 0.3882979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4607191, upper bound: 0.4692252
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4643910, upper bound: 0.4698704
time: 1.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1808972, 0.1257179, -0.1957917, 0.1376383, -0.3185356, 0.3215096
1: -0.1704915, 0.1301977, -0.1817190, 0.1446229, -0.3151144, 0.3119167
2: -0.1077426, 0.2246991, -0.1198079, 0.2389458, -0.3466884, 0.3445069
3: -0.1120043, 0.2710538, -0.1027307, 0.2922405, -0.3971365, 0.3670502
4: -0.1459320, 0.1661268, -0.1581436, 0.1856094, -0.3315413, 0.3242704
5: -0.1321877, 0.2071069, -0.1468158, 0.2227110, -0.3548986, 0.3539227
6: -0.1692593, 0.1582533, -0.1843564, 0.1744584, -0.3437178, 0.3426096
7: 0.5776505, 1.0971313, 0.5513976, 1.0772233, -0.4995728, 0.5457337
8: -0.1184425, 0.2428019, -0.1343574, 0.2585509, -0.3769934, 0.3771593
9: -0.1267133, 0.2370867, -0.1385188, 0.2523187, -0.3790320, 0.3756055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4676852, upper bound: 0.4630364
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4729467, upper bound: 0.4649783
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1808972, 0.1257179, -0.1981227, 0.1397733, -0.3206705, 0.3238406
1: -0.1704915, 0.1301977, -0.1836794, 0.1469274, -0.3174189, 0.3138772
2: -0.1077426, 0.2246991, -0.1222568, 0.2413561, -0.3490988, 0.3469559
3: -0.1120043, 0.2710538, -0.1076335, 0.2956828, -0.4007688, 0.3722016
4: -0.1459320, 0.1661268, -0.1601131, 0.1886488, -0.3345807, 0.3262398
5: -0.1321877, 0.2071069, -0.1492041, 0.2250202, -0.3572078, 0.3563110
6: -0.1692593, 0.1582533, -0.1866842, 0.1772477, -0.3465070, 0.3449374
7: 0.5776505, 1.0971313, 0.5470352, 1.0812945, -0.5036440, 0.5500960
8: -0.1184425, 0.2428019, -0.1366474, 0.2613557, -0.3797982, 0.3794493
9: -0.1267133, 0.2370867, -0.1408535, 0.2547590, -0.3814723, 0.3779402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4676852, upper bound: 0.4630364
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4729467, upper bound: 0.4649783
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1825290, 0.1272058, -0.1957917, 0.1376383, -0.3201674, 0.3229975
1: -0.1719686, 0.1321760, -0.1817190, 0.1446229, -0.3165915, 0.3138950
2: -0.1097251, 0.2265452, -0.1198079, 0.2389458, -0.3486709, 0.3463531
3: -0.1166055, 0.2735360, -0.1027307, 0.2922405, -0.4021130, 0.3695592
4: -0.1474034, 0.1682799, -0.1581436, 0.1856094, -0.3330128, 0.3264235
5: -0.1338228, 0.2090905, -0.1468158, 0.2227110, -0.3565338, 0.3559063
6: -0.1711465, 0.1601918, -0.1843564, 0.1744584, -0.3456049, 0.3445482
7: 0.5743284, 1.1006465, 0.5513976, 1.0772233, -0.5028949, 0.5492489
8: -0.1202950, 0.2449334, -0.1343574, 0.2585509, -0.3788459, 0.3792908
9: -0.1284853, 0.2390319, -0.1385188, 0.2523187, -0.3808040, 0.3775506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4612394, upper bound: 0.4612797
time: 1.24 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667387, upper bound: 0.4638896
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1825290, 0.1272058, -0.1981227, 0.1397733, -0.3223023, 0.3253285
1: -0.1719686, 0.1321760, -0.1836794, 0.1469274, -0.3188960, 0.3158555
2: -0.1097251, 0.2265452, -0.1222568, 0.2413561, -0.3510812, 0.3488021
3: -0.1166055, 0.2735360, -0.1076335, 0.2956828, -0.4055593, 0.3745651
4: -0.1474034, 0.1682799, -0.1601131, 0.1886488, -0.3360522, 0.3283929
5: -0.1338228, 0.2090905, -0.1492041, 0.2250202, -0.3588430, 0.3582946
6: -0.1711465, 0.1601918, -0.1866842, 0.1772477, -0.3483941, 0.3468760
7: 0.5743284, 1.1006465, 0.5470352, 1.0812945, -0.5069661, 0.5536113
8: -0.1202950, 0.2449334, -0.1366474, 0.2613557, -0.3816507, 0.3815808
9: -0.1284853, 0.2390319, -0.1408535, 0.2547590, -0.3832443, 0.3798854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4612394, upper bound: 0.4612797
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667387, upper bound: 0.4638896
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1808972, 0.1257179, -0.1978467, 0.1396426, -0.3205398, 0.3235646
1: -0.1704915, 0.1301977, -0.1834427, 0.1464607, -0.3169521, 0.3136404
2: -0.1077426, 0.2246991, -0.1221327, 0.2412469, -0.3489895, 0.3468318
3: -0.1120043, 0.2710538, -0.1084894, 0.2939796, -0.3988481, 0.3726371
4: -0.1459320, 0.1661268, -0.1598459, 0.1883272, -0.3342592, 0.3259726
5: -0.1321877, 0.2071069, -0.1488501, 0.2247693, -0.3569570, 0.3559571
6: -0.1692593, 0.1582533, -0.1863315, 0.1770898, -0.3463492, 0.3445847
7: 0.5776505, 1.0971313, 0.5489333, 1.0838127, -0.5061622, 0.5481980
8: -0.1184425, 0.2428019, -0.1363257, 0.2612974, -0.3797399, 0.3791277
9: -0.1267133, 0.2370867, -0.1407588, 0.2545699, -0.3812832, 0.3778455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4705914, upper bound: 0.4630138
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4759062, upper bound: 0.4649781
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1808972, 0.1257179, -0.2004130, 0.1419846, -0.3228818, 0.3261309
1: -0.1704915, 0.1301977, -0.1855939, 0.1490210, -0.3195125, 0.3157917
2: -0.1077426, 0.2246991, -0.1247960, 0.2438776, -0.3516202, 0.3494951
3: -0.1120043, 0.2710538, -0.1135312, 0.2978688, -0.4028581, 0.3779178
4: -0.1459320, 0.1661268, -0.1620215, 0.1916837, -0.3376156, 0.3281483
5: -0.1321877, 0.2071069, -0.1514867, 0.2273217, -0.3595094, 0.3585936
6: -0.1692593, 0.1582533, -0.1889065, 0.1801380, -0.3493974, 0.3471597
7: 0.5776505, 1.0971313, 0.5440536, 1.0879042, -0.5102537, 0.5530777
8: -0.1184425, 0.2428019, -0.1388672, 0.2643468, -0.3827893, 0.3816691
9: -0.1267133, 0.2370867, -0.1433101, 0.2572468, -0.3839601, 0.3803968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4705914, upper bound: 0.4630138
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4759062, upper bound: 0.4649781
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1825290, 0.1272058, -0.1978467, 0.1396426, -0.3221716, 0.3250525
1: -0.1719686, 0.1321760, -0.1834427, 0.1464607, -0.3184293, 0.3156187
2: -0.1097251, 0.2265452, -0.1221327, 0.2412469, -0.3509720, 0.3486780
3: -0.1166055, 0.2735360, -0.1084894, 0.2939796, -0.4038246, 0.3751462
4: -0.1474034, 0.1682799, -0.1598459, 0.1883272, -0.3357306, 0.3281257
5: -0.1338228, 0.2090905, -0.1488501, 0.2247693, -0.3585921, 0.3579406
6: -0.1711465, 0.1601918, -0.1863315, 0.1770898, -0.3482363, 0.3465232
7: 0.5743284, 1.1006465, 0.5489333, 1.0838127, -0.5094843, 0.5517132
8: -0.1202950, 0.2449334, -0.1363257, 0.2612974, -0.3815925, 0.3812591
9: -0.1284853, 0.2390319, -0.1407588, 0.2545699, -0.3830552, 0.3797907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4643592, upper bound: 0.4612340
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4698704, upper bound: 0.4638896
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1825290, 0.1272058, -0.2004130, 0.1419846, -0.3245136, 0.3276188
1: -0.1719686, 0.1321760, -0.1855939, 0.1490210, -0.3209896, 0.3177700
2: -0.1097251, 0.2265452, -0.1247960, 0.2438776, -0.3536026, 0.3513412
3: -0.1166055, 0.2735360, -0.1135312, 0.2978688, -0.4076569, 0.3802789
4: -0.1474034, 0.1682799, -0.1620215, 0.1916837, -0.3390871, 0.3303014
5: -0.1338228, 0.2090905, -0.1514867, 0.2273217, -0.3611445, 0.3605772
6: -0.1711465, 0.1601918, -0.1889065, 0.1801380, -0.3512845, 0.3490983
7: 0.5743284, 1.1006465, 0.5440536, 1.0879042, -0.5135758, 0.5565929
8: -0.1202950, 0.2449334, -0.1388672, 0.2643468, -0.3846418, 0.3838006
9: -0.1284853, 0.2390319, -0.1433101, 0.2572468, -0.3857321, 0.3823420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4643592, upper bound: 0.4612340
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4698704, upper bound: 0.4638896
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1829757, 0.1277481, -0.1957917, 0.1376383, -0.3206141, 0.3235398
1: -0.1724627, 0.1327230, -0.1817190, 0.1446229, -0.3170856, 0.3144420
2: -0.1104584, 0.2272168, -0.1198079, 0.2389458, -0.3494042, 0.3470247
3: -0.1184711, 0.2740996, -0.1027307, 0.2922405, -0.4036006, 0.3702002
4: -0.1478177, 0.1689606, -0.1581436, 0.1856094, -0.3334270, 0.3271042
5: -0.1343182, 0.2096441, -0.1468158, 0.2227110, -0.3570291, 0.3564599
6: -0.1716500, 0.1609033, -0.1843564, 0.1744584, -0.3461084, 0.3452597
7: 0.5735836, 1.1043050, 0.5513976, 1.0772233, -0.5036397, 0.5529075
8: -0.1208325, 0.2457149, -0.1343574, 0.2585509, -0.3793834, 0.3800723
9: -0.1291714, 0.2396820, -0.1385188, 0.2523187, -0.3814901, 0.3782007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4671754, upper bound: 0.4656954
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4717157, upper bound: 0.4674272
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1829757, 0.1277481, -0.1981227, 0.1397733, -0.3227490, 0.3258708
1: -0.1724627, 0.1327230, -0.1836794, 0.1469274, -0.3193901, 0.3164025
2: -0.1104584, 0.2272168, -0.1222568, 0.2413561, -0.3518146, 0.3494736
3: -0.1184711, 0.2740996, -0.1076335, 0.2956828, -0.4072330, 0.3753516
4: -0.1478177, 0.1689606, -0.1601131, 0.1886488, -0.3364664, 0.3290737
5: -0.1343182, 0.2096441, -0.1492041, 0.2250202, -0.3593383, 0.3588482
6: -0.1716500, 0.1609033, -0.1866842, 0.1772477, -0.3488977, 0.3475875
7: 0.5735836, 1.1043050, 0.5470352, 1.0812945, -0.5077109, 0.5572698
8: -0.1208325, 0.2457149, -0.1366474, 0.2613557, -0.3821883, 0.3823624
9: -0.1291714, 0.2396820, -0.1408535, 0.2547590, -0.3839304, 0.3805355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4671754, upper bound: 0.4656954
time: 1.10 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4717157, upper bound: 0.4674272
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1847818, 0.1293581, -0.1957917, 0.1376383, -0.3224202, 0.3251498
1: -0.1740662, 0.1348209, -0.1817190, 0.1446229, -0.3186890, 0.3165399
2: -0.1125725, 0.2292156, -0.1198079, 0.2389458, -0.3515183, 0.3490235
3: -0.1230852, 0.2766161, -0.1027307, 0.2922405, -0.4085245, 0.3727831
4: -0.1494231, 0.1712954, -0.1581436, 0.1856094, -0.3350325, 0.3294390
5: -0.1360951, 0.2117969, -0.1468158, 0.2227110, -0.3588061, 0.3586127
6: -0.1736884, 0.1630063, -0.1843564, 0.1744584, -0.3481468, 0.3473627
7: 0.5702533, 1.1078570, 0.5513976, 1.0772233, -0.5069700, 0.5564594
8: -0.1228552, 0.2480431, -0.1343574, 0.2585509, -0.3814061, 0.3824005
9: -0.1310511, 0.2418063, -0.1385188, 0.2523187, -0.3833697, 0.3803250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4612394, upper bound: 0.4645572
time: 1.26 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667387, upper bound: 0.4662509
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1847818, 0.1293581, -0.1981227, 0.1397733, -0.3245552, 0.3274808
1: -0.1740662, 0.1348209, -0.1836794, 0.1469274, -0.3209936, 0.3185004
2: -0.1125725, 0.2292156, -0.1222568, 0.2413561, -0.3539286, 0.3514724
3: -0.1230852, 0.2766161, -0.1076335, 0.2956828, -0.4119829, 0.3777838
4: -0.1494231, 0.1712954, -0.1601131, 0.1886488, -0.3380719, 0.3314084
5: -0.1360951, 0.2117969, -0.1492041, 0.2250202, -0.3611153, 0.3610010
6: -0.1736884, 0.1630063, -0.1866842, 0.1772477, -0.3509361, 0.3496905
7: 0.5702533, 1.1078570, 0.5470352, 1.0812945, -0.5110413, 0.5608218
8: -0.1228552, 0.2480431, -0.1366474, 0.2613557, -0.3842109, 0.3846906
9: -0.1310511, 0.2418063, -0.1408535, 0.2547590, -0.3858101, 0.3826598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4612394, upper bound: 0.4645572
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667387, upper bound: 0.4662509
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1829757, 0.1277481, -0.1978467, 0.1396426, -0.3226184, 0.3255948
1: -0.1724627, 0.1327230, -0.1834427, 0.1464607, -0.3189234, 0.3161657
2: -0.1104584, 0.2272168, -0.1221327, 0.2412469, -0.3517053, 0.3493495
3: -0.1184711, 0.2740996, -0.1084894, 0.2939796, -0.4052352, 0.3757029
4: -0.1478177, 0.1689606, -0.1598459, 0.1883272, -0.3361449, 0.3288065
5: -0.1343182, 0.2096441, -0.1488501, 0.2247693, -0.3590874, 0.3584942
6: -0.1716500, 0.1609033, -0.1863315, 0.1770898, -0.3487399, 0.3472348
7: 0.5735836, 1.1043050, 0.5489333, 1.0838127, -0.5102291, 0.5553718
8: -0.1208325, 0.2457149, -0.1363257, 0.2612974, -0.3821300, 0.3820407
9: -0.1291714, 0.2396820, -0.1407588, 0.2545699, -0.3837413, 0.3804408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4671755, upper bound: 0.4657179
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4717173, upper bound: 0.4674973
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1829757, 0.1277481, -0.2004130, 0.1419846, -0.3249603, 0.3281611
1: -0.1724627, 0.1327230, -0.1855939, 0.1490210, -0.3214837, 0.3183170
2: -0.1104584, 0.2272168, -0.1247960, 0.2438776, -0.3543360, 0.3520128
3: -0.1184711, 0.2740996, -0.1135312, 0.2978688, -0.4092450, 0.3809857
4: -0.1478177, 0.1689606, -0.1620215, 0.1916837, -0.3395013, 0.3309821
5: -0.1343182, 0.2096441, -0.1514867, 0.2273217, -0.3616399, 0.3611308
6: -0.1716500, 0.1609033, -0.1889065, 0.1801380, -0.3517880, 0.3498098
7: 0.5735836, 1.1043050, 0.5440536, 1.0879042, -0.5143206, 0.5602515
8: -0.1208325, 0.2457149, -0.1388672, 0.2643468, -0.3851793, 0.3845821
9: -0.1291714, 0.2396820, -0.1433101, 0.2572468, -0.3864182, 0.3829921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4671755, upper bound: 0.4657179
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4717173, upper bound: 0.4674973
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1847818, 0.1293581, -0.1978467, 0.1396426, -0.3244244, 0.3272047
1: -0.1740662, 0.1348209, -0.1834427, 0.1464607, -0.3205268, 0.3182636
2: -0.1125725, 0.2292156, -0.1221327, 0.2412469, -0.3538194, 0.3513483
3: -0.1230852, 0.2766161, -0.1084894, 0.2939796, -0.4101648, 0.3782906
4: -0.1494231, 0.1712954, -0.1598459, 0.1883272, -0.3377503, 0.3311412
5: -0.1360951, 0.2117969, -0.1488501, 0.2247693, -0.3608644, 0.3606470
6: -0.1736884, 0.1630063, -0.1863315, 0.1770898, -0.3507783, 0.3493378
7: 0.5702533, 1.1078570, 0.5489333, 1.0838127, -0.5135595, 0.5589237
8: -0.1228552, 0.2480431, -0.1363257, 0.2612974, -0.3841526, 0.3843689
9: -0.1310511, 0.2418063, -0.1407588, 0.2545699, -0.3856210, 0.3825651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4614115, upper bound: 0.4646070
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4668974, upper bound: 0.4663738
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1847818, 0.1293581, -0.2004130, 0.1419846, -0.3267664, 0.3297710
1: -0.1740662, 0.1348209, -0.1855939, 0.1490210, -0.3230872, 0.3204149
2: -0.1125725, 0.2292156, -0.1247960, 0.2438776, -0.3564500, 0.3540116
3: -0.1230852, 0.2766161, -0.1135312, 0.2978688, -0.4140024, 0.3834141
4: -0.1494231, 0.1712954, -0.1620215, 0.1916837, -0.3411068, 0.3333169
5: -0.1360951, 0.2117969, -0.1514867, 0.2273217, -0.3634169, 0.3632836
6: -0.1736884, 0.1630063, -0.1889065, 0.1801380, -0.3538264, 0.3519128
7: 0.5702533, 1.1078570, 0.5440536, 1.0879042, -0.5176510, 0.5638034
8: -0.1228552, 0.2480431, -0.1388672, 0.2643468, -0.3872020, 0.3869103
9: -0.1310511, 0.2418063, -0.1433101, 0.2572468, -0.3882979, 0.3851164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4614115, upper bound: 0.4646070
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4668974, upper bound: 0.4663738
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1808972, 0.1257179, -0.1808972, 0.1257179, -0.3066151, 0.3066151
1: -0.1704915, 0.1301977, -0.1704915, 0.1301977, -0.3006892, 0.3006892
2: -0.1077426, 0.2246991, -0.1077426, 0.2246991, -0.3324417, 0.3324417
3: -0.1120043, 0.2710538, -0.1120043, 0.2710538, -0.3750800, 0.3750800
4: -0.1459320, 0.1661268, -0.1459320, 0.1661268, -0.3120587, 0.3120587
5: -0.1321877, 0.2071069, -0.1321877, 0.2071069, -0.3392946, 0.3392946
6: -0.1692593, 0.1582533, -0.1692593, 0.1582533, -0.3275126, 0.3275126
7: 0.5776505, 1.0971313, 0.5776505, 1.0971313, -0.5194807, 0.5194807
8: -0.1184425, 0.2428019, -0.1184425, 0.2428019, -0.3612444, 0.3612444
9: -0.1267133, 0.2370867, -0.1267133, 0.2370867, -0.3638000, 0.3638000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4591560, upper bound: 0.4587880
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4641046, upper bound: 0.4607078
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1808972, 0.1257179, -0.1825290, 0.1272058, -0.3081030, 0.3082469
1: -0.1704915, 0.1301977, -0.1719686, 0.1321760, -0.3026675, 0.3021663
2: -0.1077426, 0.2246991, -0.1097251, 0.2265452, -0.3342879, 0.3344242
3: -0.1120043, 0.2710538, -0.1166055, 0.2735360, -0.3776520, 0.3800896
4: -0.1459320, 0.1661268, -0.1474034, 0.1682799, -0.3142118, 0.3135302
5: -0.1321877, 0.2071069, -0.1338228, 0.2090905, -0.3412781, 0.3409297
6: -0.1692593, 0.1582533, -0.1711465, 0.1601918, -0.3294511, 0.3293997
7: 0.5776505, 1.0971313, 0.5743284, 1.1006465, -0.5229959, 0.5228028
8: -0.1184425, 0.2428019, -0.1202950, 0.2449334, -0.3633759, 0.3630970
9: -0.1267133, 0.2370867, -0.1284853, 0.2390319, -0.3657452, 0.3655721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4591560, upper bound: 0.4587880
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4641046, upper bound: 0.4607078
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1825290, 0.1272058, -0.1808972, 0.1257179, -0.3082469, 0.3081030
1: -0.1719686, 0.1321760, -0.1704915, 0.1301977, -0.3021663, 0.3026675
2: -0.1097251, 0.2265452, -0.1077426, 0.2246991, -0.3344242, 0.3342879
3: -0.1166055, 0.2735360, -0.1120043, 0.2710538, -0.3800897, 0.3776521
4: -0.1474034, 0.1682799, -0.1459320, 0.1661268, -0.3135302, 0.3142118
5: -0.1338228, 0.2090905, -0.1321877, 0.2071069, -0.3409297, 0.3412781
6: -0.1711465, 0.1601918, -0.1692593, 0.1582533, -0.3293997, 0.3294511
7: 0.5743284, 1.1006465, 0.5776505, 1.0971313, -0.5228028, 0.5229959
8: -0.1202950, 0.2449334, -0.1184425, 0.2428019, -0.3630970, 0.3633759
9: -0.1284853, 0.2390319, -0.1267133, 0.2370867, -0.3655721, 0.3657452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4544854, upper bound: 0.4568578
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4597271, upper bound: 0.4597271
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1825290, 0.1272058, -0.1825290, 0.1272058, -0.3097348, 0.3097348
1: -0.1719686, 0.1321760, -0.1719686, 0.1321760, -0.3041447, 0.3041447
2: -0.1097251, 0.2265452, -0.1097251, 0.2265452, -0.3362703, 0.3362703
3: -0.1166055, 0.2735360, -0.1166055, 0.2735360, -0.3825014, 0.3825015
4: -0.1474034, 0.1682799, -0.1474034, 0.1682799, -0.3156833, 0.3156833
5: -0.1338228, 0.2090905, -0.1338228, 0.2090905, -0.3429133, 0.3429133
6: -0.1711465, 0.1601918, -0.1711465, 0.1601918, -0.3313382, 0.3313382
7: 0.5743284, 1.1006465, 0.5743284, 1.1006465, -0.5263181, 0.5263181
8: -0.1202950, 0.2449334, -0.1202950, 0.2449334, -0.3652284, 0.3652284
9: -0.1284853, 0.2390319, -0.1284853, 0.2390319, -0.3675172, 0.3675172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4544854, upper bound: 0.4568578
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4597271, upper bound: 0.4597271
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1808972, 0.1257179, -0.1829757, 0.1277481, -0.3086453, 0.3086937
1: -0.1704915, 0.1301977, -0.1724627, 0.1327230, -0.3032145, 0.3026604
2: -0.1077426, 0.2246991, -0.1104584, 0.2272168, -0.3349594, 0.3351575
3: -0.1120043, 0.2710538, -0.1184711, 0.2740996, -0.3781195, 0.3814650
4: -0.1459320, 0.1661268, -0.1478177, 0.1689606, -0.3148926, 0.3139445
5: -0.1321877, 0.2071069, -0.1343182, 0.2096441, -0.3418317, 0.3414251
6: -0.1692593, 0.1582533, -0.1716500, 0.1609033, -0.3301627, 0.3299033
7: 0.5776505, 1.0971313, 0.5735836, 1.1043050, -0.5266545, 0.5235476
8: -0.1184425, 0.2428019, -0.1208325, 0.2457149, -0.3641574, 0.3636344
9: -0.1267133, 0.2370867, -0.1291714, 0.2396820, -0.3663953, 0.3662581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4615492, upper bound: 0.4587816
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667723, upper bound: 0.4607078
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1808972, 0.1257179, -0.1847818, 0.1293581, -0.3102553, 0.3104998
1: -0.1704915, 0.1301977, -0.1740662, 0.1348209, -0.3053124, 0.3042639
2: -0.1077426, 0.2246991, -0.1125725, 0.2292156, -0.3369582, 0.3372715
3: -0.1120043, 0.2710538, -0.1230852, 0.2766161, -0.3807526, 0.3864434
4: -0.1459320, 0.1661268, -0.1494231, 0.1712954, -0.3172273, 0.3155499
5: -0.1321877, 0.2071069, -0.1360951, 0.2117969, -0.3439845, 0.3432021
6: -0.1692593, 0.1582533, -0.1736884, 0.1630063, -0.3322657, 0.3319417
7: 0.5776505, 1.0971313, 0.5702533, 1.1078570, -0.5302064, 0.5268780
8: -0.1184425, 0.2428019, -0.1228552, 0.2480431, -0.3664857, 0.3656571
9: -0.1267133, 0.2370867, -0.1310511, 0.2418063, -0.3685196, 0.3681378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4615492, upper bound: 0.4587816
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667723, upper bound: 0.4607078
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1825290, 0.1272058, -0.1829757, 0.1277481, -0.3102771, 0.3101816
1: -0.1719686, 0.1321760, -0.1724627, 0.1327230, -0.3046916, 0.3046387
2: -0.1097251, 0.2265452, -0.1104584, 0.2272168, -0.3369419, 0.3370037
3: -0.1166055, 0.2735360, -0.1184711, 0.2740996, -0.3831292, 0.3840371
4: -0.1474034, 0.1682799, -0.1478177, 0.1689606, -0.3163640, 0.3160976
5: -0.1338228, 0.2090905, -0.1343182, 0.2096441, -0.3434669, 0.3434086
6: -0.1711465, 0.1601918, -0.1716500, 0.1609033, -0.3320498, 0.3318418
7: 0.5743284, 1.1006465, 0.5735836, 1.1043050, -0.5299766, 0.5270629
8: -0.1202950, 0.2449334, -0.1208325, 0.2457149, -0.3660100, 0.3657659
9: -0.1284853, 0.2390319, -0.1291714, 0.2396820, -0.3681673, 0.3682033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4568949, upper bound: 0.4568144
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4626156, upper bound: 0.4597271
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1825290, 0.1272058, -0.1847818, 0.1293581, -0.3118871, 0.3119877
1: -0.1719686, 0.1321760, -0.1740662, 0.1348209, -0.3067895, 0.3062422
2: -0.1097251, 0.2265452, -0.1125725, 0.2292156, -0.3389407, 0.3391177
3: -0.1166055, 0.2735360, -0.1230852, 0.2766161, -0.3856083, 0.3888720
4: -0.1474034, 0.1682799, -0.1494231, 0.1712954, -0.3186988, 0.3177030
5: -0.1338228, 0.2090905, -0.1360951, 0.2117969, -0.3456197, 0.3451856
6: -0.1711465, 0.1601918, -0.1736884, 0.1630063, -0.3341528, 0.3338802
7: 0.5743284, 1.1006465, 0.5702533, 1.1078570, -0.5335286, 0.5303932
8: -0.1202950, 0.2449334, -0.1228552, 0.2480431, -0.3683382, 0.3677886
9: -0.1284853, 0.2390319, -0.1310511, 0.2418063, -0.3702916, 0.3700829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4568949, upper bound: 0.4568144
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4626156, upper bound: 0.4597271
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1829757, 0.1277481, -0.1808972, 0.1257179, -0.3086937, 0.3086453
1: -0.1724627, 0.1327230, -0.1704915, 0.1301977, -0.3026604, 0.3032145
2: -0.1104584, 0.2272168, -0.1077426, 0.2246991, -0.3351575, 0.3349594
3: -0.1184711, 0.2740996, -0.1120043, 0.2710538, -0.3814649, 0.3781196
4: -0.1478177, 0.1689606, -0.1459320, 0.1661268, -0.3139445, 0.3148926
5: -0.1343182, 0.2096441, -0.1321877, 0.2071069, -0.3414251, 0.3418317
6: -0.1716500, 0.1609033, -0.1692593, 0.1582533, -0.3299033, 0.3301627
7: 0.5735836, 1.1043050, 0.5776505, 1.0971313, -0.5235476, 0.5266545
8: -0.1208325, 0.2457149, -0.1184425, 0.2428019, -0.3636344, 0.3641574
9: -0.1291714, 0.2396820, -0.1267133, 0.2370867, -0.3662581, 0.3663953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4590279, upper bound: 0.4617139
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4637131, upper bound: 0.4634685
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1829757, 0.1277481, -0.1825290, 0.1272058, -0.3101816, 0.3102771
1: -0.1724627, 0.1327230, -0.1719686, 0.1321760, -0.3046387, 0.3046916
2: -0.1104584, 0.2272168, -0.1097251, 0.2265452, -0.3370037, 0.3369419
3: -0.1184711, 0.2740996, -0.1166055, 0.2735360, -0.3840371, 0.3831292
4: -0.1478177, 0.1689606, -0.1474034, 0.1682799, -0.3160976, 0.3163640
5: -0.1343182, 0.2096441, -0.1338228, 0.2090905, -0.3434086, 0.3434669
6: -0.1716500, 0.1609033, -0.1711465, 0.1601918, -0.3318418, 0.3320498
7: 0.5735836, 1.1043050, 0.5743284, 1.1006465, -0.5270629, 0.5299766
8: -0.1208325, 0.2457149, -0.1202950, 0.2449334, -0.3657659, 0.3660100
9: -0.1291714, 0.2396820, -0.1284853, 0.2390319, -0.3682033, 0.3681673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4590279, upper bound: 0.4617139
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4637131, upper bound: 0.4634685
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1847818, 0.1293581, -0.1808972, 0.1257179, -0.3104998, 0.3102553
1: -0.1740662, 0.1348209, -0.1704915, 0.1301977, -0.3042639, 0.3053124
2: -0.1125725, 0.2292156, -0.1077426, 0.2246991, -0.3372715, 0.3369582
3: -0.1230852, 0.2766161, -0.1120043, 0.2710538, -0.3864434, 0.3807525
4: -0.1494231, 0.1712954, -0.1459320, 0.1661268, -0.3155499, 0.3172273
5: -0.1360951, 0.2117969, -0.1321877, 0.2071069, -0.3432021, 0.3439845
6: -0.1736884, 0.1630063, -0.1692593, 0.1582533, -0.3319417, 0.3322657
7: 0.5702533, 1.1078570, 0.5776505, 1.0971313, -0.5268780, 0.5302064
8: -0.1228552, 0.2480431, -0.1184425, 0.2428019, -0.3656571, 0.3664857
9: -0.1310511, 0.2418063, -0.1267133, 0.2370867, -0.3681378, 0.3685196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4544854, upper bound: 0.4605145
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4597271, upper bound: 0.4626156
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1847818, 0.1293581, -0.1825290, 0.1272058, -0.3119877, 0.3118871
1: -0.1740662, 0.1348209, -0.1719686, 0.1321760, -0.3062422, 0.3067895
2: -0.1125725, 0.2292156, -0.1097251, 0.2265452, -0.3391177, 0.3389407
3: -0.1230852, 0.2766161, -0.1166055, 0.2735360, -0.3888720, 0.3856083
4: -0.1494231, 0.1712954, -0.1474034, 0.1682799, -0.3177030, 0.3186988
5: -0.1360951, 0.2117969, -0.1338228, 0.2090905, -0.3451856, 0.3456197
6: -0.1736884, 0.1630063, -0.1711465, 0.1601918, -0.3338802, 0.3341528
7: 0.5702533, 1.1078570, 0.5743284, 1.1006465, -0.5303932, 0.5335286
8: -0.1228552, 0.2480431, -0.1202950, 0.2449334, -0.3677886, 0.3683382
9: -0.1310511, 0.2418063, -0.1284853, 0.2390319, -0.3700829, 0.3702916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4544854, upper bound: 0.4605145
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4597271, upper bound: 0.4626156
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1829757, 0.1277481, -0.1829757, 0.1277481, -0.3107239, 0.3107239
1: -0.1724627, 0.1327230, -0.1724627, 0.1327230, -0.3051857, 0.3051857
2: -0.1104584, 0.2272168, -0.1104584, 0.2272168, -0.3376752, 0.3376752
3: -0.1184711, 0.2740996, -0.1184711, 0.2740996, -0.3844112, 0.3844112
4: -0.1478177, 0.1689606, -0.1478177, 0.1689606, -0.3167783, 0.3167783
5: -0.1343182, 0.2096441, -0.1343182, 0.2096441, -0.3439623, 0.3439623
6: -0.1716500, 0.1609033, -0.1716500, 0.1609033, -0.3325534, 0.3325534
7: 0.5735836, 1.1043050, 0.5735836, 1.1043050, -0.5307214, 0.5307214
8: -0.1208325, 0.2457149, -0.1208325, 0.2457149, -0.3665475, 0.3665475
9: -0.1291714, 0.2396820, -0.1291714, 0.2396820, -0.3688534, 0.3688534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4592096, upper bound: 0.4617158
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4637573, upper bound: 0.4635120
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1829757, 0.1277481, -0.1847818, 0.1293581, -0.3123338, 0.3125300
1: -0.1724627, 0.1327230, -0.1740662, 0.1348209, -0.3072836, 0.3067892
2: -0.1104584, 0.2272168, -0.1125725, 0.2292156, -0.3396740, 0.3397893
3: -0.1184711, 0.2740996, -0.1230852, 0.2766161, -0.3870480, 0.3893942
4: -0.1478177, 0.1689606, -0.1494231, 0.1712954, -0.3191130, 0.3183837
5: -0.1343182, 0.2096441, -0.1360951, 0.2117969, -0.3461151, 0.3457392
6: -0.1716500, 0.1609033, -0.1736884, 0.1630063, -0.3346564, 0.3345917
7: 0.5735836, 1.1043050, 0.5702533, 1.1078570, -0.5342734, 0.5340518
8: -0.1208325, 0.2457149, -0.1228552, 0.2480431, -0.3688757, 0.3685701
9: -0.1291714, 0.2396820, -0.1310511, 0.2418063, -0.3709777, 0.3707330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4592096, upper bound: 0.4617158
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4637573, upper bound: 0.4635120
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1847818, 0.1293581, -0.1829757, 0.1277481, -0.3125300, 0.3123338
1: -0.1740662, 0.1348209, -0.1724627, 0.1327230, -0.3067892, 0.3072836
2: -0.1125725, 0.2292156, -0.1104584, 0.2272168, -0.3397893, 0.3396740
3: -0.1230852, 0.2766161, -0.1184711, 0.2740996, -0.3893943, 0.3870479
4: -0.1494231, 0.1712954, -0.1478177, 0.1689606, -0.3183837, 0.3191130
5: -0.1360951, 0.2117969, -0.1343182, 0.2096441, -0.3457392, 0.3461151
6: -0.1736884, 0.1630063, -0.1716500, 0.1609033, -0.3345917, 0.3346564
7: 0.5702533, 1.1078570, 0.5735836, 1.1043050, -0.5340518, 0.5342734
8: -0.1228552, 0.2480431, -0.1208325, 0.2457149, -0.3685701, 0.3688757
9: -0.1310511, 0.2418063, -0.1291714, 0.2396820, -0.3707330, 0.3709777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4549400, upper bound: 0.4605145
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4603877, upper bound: 0.4626874
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1847818, 0.1293581, -0.1847818, 0.1293581, -0.3141399, 0.3141399
1: -0.1740662, 0.1348209, -0.1740662, 0.1348209, -0.3088871, 0.3088871
2: -0.1125725, 0.2292156, -0.1125725, 0.2292156, -0.3417881, 0.3417881
3: -0.1230852, 0.2766161, -0.1230852, 0.2766161, -0.3918891, 0.3918891
4: -0.1494231, 0.1712954, -0.1494231, 0.1712954, -0.3207185, 0.3207185
5: -0.1360951, 0.2117969, -0.1360951, 0.2117969, -0.3478920, 0.3478920
6: -0.1736884, 0.1630063, -0.1736884, 0.1630063, -0.3366947, 0.3366947
7: 0.5702533, 1.1078570, 0.5702533, 1.1078570, -0.5376037, 0.5376037
8: -0.1228552, 0.2480431, -0.1228552, 0.2480431, -0.3708983, 0.3708983
9: -0.1310511, 0.2418063, -0.1310511, 0.2418063, -0.3728573, 0.3728573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4549400, upper bound: 0.4605145
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4603877, upper bound: 0.4626874
time: 1.04 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.77 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4791408, upper bound: 0.4792314
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4819819, upper bound: 0.4804261
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4791408, upper bound: 0.4792314
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4819819, upper bound: 0.4804261
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4776461, upper bound: 0.4788126
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4801322, upper bound: 0.4801323
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4776461, upper bound: 0.4788126
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4801322, upper bound: 0.4801323
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4836484, upper bound: 0.4788719
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4866729, upper bound: 0.4801987
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4836484, upper bound: 0.4788719
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4866729, upper bound: 0.4801987
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4818949, upper bound: 0.4783912
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4846592, upper bound: 0.4798247
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4818949, upper bound: 0.4783912
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4846592, upper bound: 0.4798247
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4788160, upper bound: 0.4839497
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4812841, upper bound: 0.4847832
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4788160, upper bound: 0.4839497
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4812841, upper bound: 0.4847832
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4774562, upper bound: 0.4839114
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4798247, upper bound: 0.4846592
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4774562, upper bound: 0.4839114
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4798247, upper bound: 0.4846592
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4793078, upper bound: 0.4840348
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4817020, upper bound: 0.4848903
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4793078, upper bound: 0.4840348
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4817020, upper bound: 0.4848903
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4781874, upper bound: 0.4840080
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4805440, upper bound: 0.4848034
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4781874, upper bound: 0.4840080
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4805440, upper bound: 0.4848034
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4626142, upper bound: 0.4662597
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4664864, upper bound: 0.4672148
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4626142, upper bound: 0.4662597
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4664864, upper bound: 0.4672148
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4601998, upper bound: 0.4654807
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4638896, upper bound: 0.4667387
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4601998, upper bound: 0.4654807
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4638896, upper bound: 0.4667387
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4645760, upper bound: 0.4662597
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4686553, upper bound: 0.4672148
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4645760, upper bound: 0.4662597
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4686553, upper bound: 0.4672148
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4626839, upper bound: 0.4654708
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4662509, upper bound: 0.4667387
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4626839, upper bound: 0.4654708
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4662509, upper bound: 0.4667387
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4626142, upper bound: 0.4693404
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4664144, upper bound: 0.4699818
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4626142, upper bound: 0.4693404
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4664144, upper bound: 0.4699818
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4601996, upper bound: 0.4692252
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4638896, upper bound: 0.4698704
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4601996, upper bound: 0.4692252
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4638896, upper bound: 0.4698704
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4628442, upper bound: 0.4693404
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4665851, upper bound: 0.4699818
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4628442, upper bound: 0.4693404
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4665851, upper bound: 0.4699818
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4607191, upper bound: 0.4692252
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4643910, upper bound: 0.4698704
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4607191, upper bound: 0.4692252
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4643910, upper bound: 0.4698704
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4676852, upper bound: 0.4630364
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4729467, upper bound: 0.4649783
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4676852, upper bound: 0.4630364
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4729467, upper bound: 0.4649783
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4612394, upper bound: 0.4612797
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4667387, upper bound: 0.4638896
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4612394, upper bound: 0.4612797
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4667387, upper bound: 0.4638896
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4705914, upper bound: 0.4630138
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4759062, upper bound: 0.4649781
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4705914, upper bound: 0.4630138
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4759062, upper bound: 0.4649781
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4643592, upper bound: 0.4612340
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4698704, upper bound: 0.4638896
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4643592, upper bound: 0.4612340
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4698704, upper bound: 0.4638896
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4671754, upper bound: 0.4656954
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4717157, upper bound: 0.4674272
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4671754, upper bound: 0.4656954
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4717157, upper bound: 0.4674272
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4612394, upper bound: 0.4645572
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4667387, upper bound: 0.4662509
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4612394, upper bound: 0.4645572
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4667387, upper bound: 0.4662509
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4671755, upper bound: 0.4657179
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4717173, upper bound: 0.4674973
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4671755, upper bound: 0.4657179
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4717173, upper bound: 0.4674973
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4614115, upper bound: 0.4646070
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4668974, upper bound: 0.4663738
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4614115, upper bound: 0.4646070
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4668974, upper bound: 0.4663738
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4591560, upper bound: 0.4587880
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4641046, upper bound: 0.4607078
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4591560, upper bound: 0.4587880
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4641046, upper bound: 0.4607078
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4544854, upper bound: 0.4568578
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4597271, upper bound: 0.4597271
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4544854, upper bound: 0.4568578
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4597271, upper bound: 0.4597271
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4615492, upper bound: 0.4587816
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4667723, upper bound: 0.4607078
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4615492, upper bound: 0.4587816
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4667723, upper bound: 0.4607078
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4568949, upper bound: 0.4568144
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4626156, upper bound: 0.4597271
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4568949, upper bound: 0.4568144
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4626156, upper bound: 0.4597271
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4590279, upper bound: 0.4617139
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4637131, upper bound: 0.4634685
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4590279, upper bound: 0.4617139
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4637131, upper bound: 0.4634685
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4544854, upper bound: 0.4605145
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4597271, upper bound: 0.4626156
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4544854, upper bound: 0.4605145
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4597271, upper bound: 0.4626156
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4592096, upper bound: 0.4617158
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4637573, upper bound: 0.4635120
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4592096, upper bound: 0.4617158
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4637573, upper bound: 0.4635120
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4549400, upper bound: 0.4605145
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4603877, upper bound: 0.4626874
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4549400, upper bound: 0.4605145
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 7, lower bound: -0.4603877, upper bound: 0.4626874

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1798170, 0.1249073, -0.1938129, 0.1358960, -0.3157130, 0.3187202
1: -0.1678406, 0.1251325, -0.1800087, 0.1424643, -0.3103049, 0.3051412
2: -0.1030130, 0.2220117, -0.1178163, 0.2369544, -0.3399674, 0.3398280
3: -0.0835202, 0.2567468, -0.1007997, 0.2881706, -0.3637855, 0.3511781
4: -0.1447961, 0.1643858, -0.1564491, 0.1830170, -0.3278131, 0.3208349
5: -0.1298840, 0.2061689, -0.1446947, 0.2207694, -0.3506534, 0.3508636
6: -0.1676237, 0.1551396, -0.1823366, 0.1721205, -0.3397442, 0.3374762
7: 0.5965656, 1.0654389, 0.5563824, 1.0760221, -0.4794565, 0.5090565
8: -0.1178138, 0.2404614, -0.1323600, 0.2563326, -0.3741465, 0.3728214
9: -0.1221031, 0.2353750, -0.1365970, 0.2502758, -0.3723789, 0.3719720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4805930, upper bound: 0.4805930
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4805930, upper bound: 0.4823533
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1873402, 0.1305287, -0.1957917, 0.1376383, -0.3249785, 0.3263204
1: -0.1744099, 0.1349606, -0.1817190, 0.1446229, -0.3190327, 0.3166796
2: -0.1111550, 0.2302768, -0.1198079, 0.2389458, -0.3501008, 0.3500847
3: -0.0934817, 0.2739309, -0.1027307, 0.2922405, -0.3788756, 0.3702467
4: -0.1510423, 0.1744190, -0.1581436, 0.1856094, -0.3366517, 0.3325626
5: -0.1377659, 0.2143259, -0.1468158, 0.2227110, -0.3604769, 0.3611417
6: -0.1757190, 0.1642851, -0.1843564, 0.1744584, -0.3501774, 0.3486415
7: 0.5742629, 1.0715764, 0.5513976, 1.0772233, -0.5029604, 0.5201788
8: -0.1258322, 0.2490739, -0.1343574, 0.2585509, -0.3843830, 0.3834313
9: -0.1300584, 0.2436127, -0.1385188, 0.2523187, -0.3823770, 0.3821315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4823533, upper bound: 0.4807703
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4823533, upper bound: 0.4841629
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1798170, 0.1249073, -0.1956325, 0.1375789, -0.3173959, 0.3205398
1: -0.1678406, 0.1251325, -0.1815816, 0.1443198, -0.3121603, 0.3067141
2: -0.1030130, 0.2220117, -0.1198278, 0.2388553, -0.3418683, 0.3418395
3: -0.0835202, 0.2567468, -0.1056831, 0.2910315, -0.3668495, 0.3562868
4: -0.1447961, 0.1643858, -0.1579999, 0.1853769, -0.3301730, 0.3223857
5: -0.1298840, 0.2061689, -0.1465750, 0.2225713, -0.3524552, 0.3527439
6: -0.1676237, 0.1551396, -0.1841727, 0.1743157, -0.3419393, 0.3393123
7: 0.5965656, 1.0654389, 0.5527422, 1.0800741, -0.4835085, 0.5126966
8: -0.1178138, 0.2404614, -0.1341306, 0.2585621, -0.3763759, 0.3745920
9: -0.1221031, 0.2353750, -0.1384891, 0.2521901, -0.3742932, 0.3738641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4788620, upper bound: 0.4778063
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4788620, upper bound: 0.4792314
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1873402, 0.1305287, -0.1981227, 0.1397733, -0.3271135, 0.3286514
1: -0.1744099, 0.1349606, -0.1836794, 0.1469274, -0.3213373, 0.3186400
2: -0.1111550, 0.2302768, -0.1222568, 0.2413561, -0.3525111, 0.3525336
3: -0.0934817, 0.2739309, -0.1076335, 0.2956828, -0.3825080, 0.3754003
4: -0.1510423, 0.1744190, -0.1601131, 0.1886488, -0.3396911, 0.3345320
5: -0.1377659, 0.2143259, -0.1492041, 0.2250202, -0.3627861, 0.3635300
6: -0.1757190, 0.1642851, -0.1866842, 0.1772477, -0.3529667, 0.3509693
7: 0.5742629, 1.0715764, 0.5470352, 1.0812945, -0.5070317, 0.5245411
8: -0.1258322, 0.2490739, -0.1366474, 0.2613557, -0.3871879, 0.3857213
9: -0.1300584, 0.2436127, -0.1408535, 0.2547590, -0.3848174, 0.3844662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4802415, upper bound: 0.4779194
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4802415, upper bound: 0.4804261
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1812438, 0.1262368, -0.1938129, 0.1358960, -0.3171398, 0.3200496
1: -0.1692075, 0.1270077, -0.1800087, 0.1424643, -0.3116718, 0.3070165
2: -0.1048302, 0.2237196, -0.1178163, 0.2369544, -0.3417845, 0.3415359
3: -0.0883701, 0.2593294, -0.1007997, 0.2881706, -0.3686938, 0.3539520
4: -0.1461221, 0.1663492, -0.1564491, 0.1830170, -0.3291391, 0.3227983
5: -0.1313906, 0.2079459, -0.1446947, 0.2207694, -0.3521600, 0.3526406
6: -0.1693185, 0.1569325, -0.1823366, 0.1721205, -0.3414390, 0.3392690
7: 0.5932113, 1.0695045, 0.5563824, 1.0760221, -0.4828109, 0.5131221
8: -0.1194964, 0.2423275, -0.1323600, 0.2563326, -0.3758290, 0.3746875
9: -0.1237533, 0.2371575, -0.1365970, 0.2502758, -0.3740291, 0.3737545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4778063, upper bound: 0.4788620
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4778063, upper bound: 0.4802415
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1888432, 0.1317277, -0.1957917, 0.1376383, -0.3264815, 0.3275194
1: -0.1756330, 0.1366000, -0.1817190, 0.1446229, -0.3202559, 0.3183191
2: -0.1128298, 0.2319590, -0.1198079, 0.2389458, -0.3517756, 0.3517669
3: -0.0983431, 0.2761314, -0.1027307, 0.2922405, -0.3838766, 0.3726481
4: -0.1522047, 0.1764771, -0.1581436, 0.1856094, -0.3378141, 0.3346208
5: -0.1392468, 0.2158999, -0.1468158, 0.2227110, -0.3619578, 0.3627157
6: -0.1772040, 0.1662185, -0.1843564, 0.1744584, -0.3516624, 0.3505749
7: 0.5710945, 1.0755857, 0.5513976, 1.0772233, -0.5061288, 0.5241882
8: -0.1272983, 0.2509542, -0.1343574, 0.2585509, -0.3858492, 0.3853115
9: -0.1317314, 0.2451840, -0.1385188, 0.2523187, -0.3840500, 0.3837028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4792314, upper bound: 0.4791408
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4792314, upper bound: 0.4819819
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1812438, 0.1262368, -0.1956325, 0.1375789, -0.3188227, 0.3218693
1: -0.1692075, 0.1270077, -0.1815816, 0.1443198, -0.3135273, 0.3085894
2: -0.1048302, 0.2237196, -0.1198278, 0.2388553, -0.3436854, 0.3435474
3: -0.0883701, 0.2593294, -0.1056831, 0.2910315, -0.3714941, 0.3588606
4: -0.1461221, 0.1663492, -0.1579999, 0.1853769, -0.3314990, 0.3243492
5: -0.1313906, 0.2079459, -0.1465750, 0.2225713, -0.3539619, 0.3545209
6: -0.1693185, 0.1569325, -0.1841727, 0.1743157, -0.3436341, 0.3411051
7: 0.5932113, 1.0695045, 0.5527422, 1.0800741, -0.4868628, 0.5167623
8: -0.1194964, 0.2423275, -0.1341306, 0.2585621, -0.3780585, 0.3764580
9: -0.1237533, 0.2371575, -0.1384891, 0.2521901, -0.3759435, 0.3756465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4774626, upper bound: 0.4774626
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4774626, upper bound: 0.4788126
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1888432, 0.1317277, -0.1981227, 0.1397733, -0.3286165, 0.3298504
1: -0.1756330, 0.1366000, -0.1836794, 0.1469274, -0.3225604, 0.3202795
2: -0.1128298, 0.2319590, -0.1222568, 0.2413561, -0.3541859, 0.3542158
3: -0.0983431, 0.2761314, -0.1076335, 0.2956828, -0.3873090, 0.3775948
4: -0.1522047, 0.1764771, -0.1601131, 0.1886488, -0.3408535, 0.3365902
5: -0.1392468, 0.2158999, -0.1492041, 0.2250202, -0.3642670, 0.3651040
6: -0.1772040, 0.1662185, -0.1866842, 0.1772477, -0.3544517, 0.3529027
7: 0.5710945, 1.0755857, 0.5470352, 1.0812945, -0.5102001, 0.5285505
8: -0.1272983, 0.2509542, -0.1366474, 0.2613557, -0.3886541, 0.3876016
9: -0.1317314, 0.2451840, -0.1408535, 0.2547590, -0.3864904, 0.3860375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4788126, upper bound: 0.4776461
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4788126, upper bound: 0.4801323
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1798170, 0.1249073, -0.1953661, 0.1374556, -0.3172726, 0.3202734
1: -0.1678406, 0.1251325, -0.1813436, 0.1438548, -0.3116954, 0.3064761
2: -0.1030130, 0.2220117, -0.1197009, 0.2387497, -0.3417627, 0.3417125
3: -0.0835202, 0.2567468, -0.1065595, 0.2893173, -0.3648901, 0.3567513
4: -0.1447961, 0.1643858, -0.1577384, 0.1850716, -0.3298677, 0.3221242
5: -0.1298840, 0.2061689, -0.1462270, 0.2223307, -0.3522147, 0.3523958
6: -0.1676237, 0.1551396, -0.1838282, 0.1741630, -0.3417866, 0.3389678
7: 0.5965656, 1.0654389, 0.5546491, 1.0825963, -0.4860307, 0.5107898
8: -0.1178138, 0.2404614, -0.1338209, 0.2585112, -0.3763250, 0.3742823
9: -0.1221031, 0.2353750, -0.1383950, 0.2520067, -0.3741098, 0.3737700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4852532, upper bound: 0.4803882
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4852532, upper bound: 0.4818313
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1873402, 0.1305287, -0.1978467, 0.1396426, -0.3269828, 0.3283754
1: -0.1744099, 0.1349606, -0.1834427, 0.1464607, -0.3208705, 0.3184032
2: -0.1111550, 0.2302768, -0.1221327, 0.2412469, -0.3524019, 0.3524095
3: -0.0934817, 0.2739309, -0.1084894, 0.2939796, -0.3805872, 0.3758382
4: -0.1510423, 0.1744190, -0.1598459, 0.1883272, -0.3393695, 0.3342648
5: -0.1377659, 0.2143259, -0.1488501, 0.2247693, -0.3625352, 0.3631760
6: -0.1757190, 0.1642851, -0.1863315, 0.1770898, -0.3528088, 0.3506165
7: 0.5742629, 1.0715764, 0.5489333, 1.0838127, -0.5095499, 0.5226431
8: -0.1258322, 0.2490739, -0.1363257, 0.2612974, -0.3871296, 0.3853996
9: -0.1300584, 0.2436127, -0.1407588, 0.2545699, -0.3846283, 0.3843715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4872231, upper bound: 0.4806286
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4872231, upper bound: 0.4838037
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1798170, 0.1249073, -0.1979244, 0.1397889, -0.3196059, 0.3228317
1: -0.1678406, 0.1251325, -0.1834880, 0.1464162, -0.3142568, 0.3086204
2: -0.1030130, 0.2220117, -0.1223550, 0.2413696, -0.3443826, 0.3443667
3: -0.0835202, 0.2567468, -0.1115895, 0.2932118, -0.3689263, 0.3620265
4: -0.1447961, 0.1643858, -0.1599081, 0.1884172, -0.3332133, 0.3242939
5: -0.1298840, 0.2061689, -0.1488572, 0.2248771, -0.3547610, 0.3550261
6: -0.1676237, 0.1551396, -0.1863988, 0.1771987, -0.3448224, 0.3415384
7: 0.5965656, 1.0654389, 0.5497698, 1.0866816, -0.4901160, 0.5156691
8: -0.1178138, 0.2404614, -0.1363553, 0.2615465, -0.3793603, 0.3768167
9: -0.1221031, 0.2353750, -0.1409385, 0.2546741, -0.3767772, 0.3763135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4836415, upper bound: 0.4776477
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4836415, upper bound: 0.4788719
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1873402, 0.1305287, -0.2004130, 0.1419846, -0.3293247, 0.3309417
1: -0.1744099, 0.1349606, -0.1855939, 0.1490210, -0.3234309, 0.3205545
2: -0.1111550, 0.2302768, -0.1247960, 0.2438776, -0.3550325, 0.3550728
3: -0.0934817, 0.2739309, -0.1135312, 0.2978688, -0.3845973, 0.3811039
4: -0.1510423, 0.1744190, -0.1620215, 0.1916837, -0.3427260, 0.3364405
5: -0.1377659, 0.2143259, -0.1514867, 0.2273217, -0.3650877, 0.3658126
6: -0.1757190, 0.1642851, -0.1889065, 0.1801380, -0.3558570, 0.3531916
7: 0.5742629, 1.0715764, 0.5440536, 1.0879042, -0.5136414, 0.5275228
8: -0.1258322, 0.2490739, -0.1388672, 0.2643468, -0.3901789, 0.3879411
9: -0.1300584, 0.2436127, -0.1433101, 0.2572468, -0.3873052, 0.3869228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4857183, upper bound: 0.4777930
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4857183, upper bound: 0.4801987
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1812438, 0.1262368, -0.1953661, 0.1374556, -0.3186994, 0.3216028
1: -0.1692075, 0.1270077, -0.1813436, 0.1438548, -0.3130624, 0.3083514
2: -0.1048302, 0.2237196, -0.1197009, 0.2387497, -0.3435798, 0.3434204
3: -0.0883701, 0.2593294, -0.1065595, 0.2893173, -0.3697985, 0.3595251
4: -0.1461221, 0.1663492, -0.1577384, 0.1850716, -0.3311937, 0.3240876
5: -0.1313906, 0.2079459, -0.1462270, 0.2223307, -0.3537214, 0.3541729
6: -0.1693185, 0.1569325, -0.1838282, 0.1741630, -0.3434814, 0.3407607
7: 0.5932113, 1.0695045, 0.5546491, 1.0825963, -0.4893850, 0.5148554
8: -0.1194964, 0.2423275, -0.1338209, 0.2585112, -0.3780076, 0.3761484
9: -0.1237533, 0.2371575, -0.1383950, 0.2520067, -0.3757600, 0.3755524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4820317, upper bound: 0.4784747
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4820317, upper bound: 0.4794660
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1888432, 0.1317277, -0.1978467, 0.1396426, -0.3284858, 0.3295743
1: -0.1756330, 0.1366000, -0.1834427, 0.1464607, -0.3220936, 0.3200427
2: -0.1128298, 0.2319590, -0.1221327, 0.2412469, -0.3540767, 0.3540917
3: -0.0983431, 0.2761314, -0.1084894, 0.2939796, -0.3855883, 0.3782396
4: -0.1522047, 0.1764771, -0.1598459, 0.1883272, -0.3405319, 0.3363230
5: -0.1392468, 0.2158999, -0.1488501, 0.2247693, -0.3640161, 0.3647500
6: -0.1772040, 0.1662185, -0.1863315, 0.1770898, -0.3542939, 0.3525500
7: 0.5710945, 1.0755857, 0.5489333, 1.0838127, -0.5127183, 0.5266525
8: -0.1272983, 0.2509542, -0.1363257, 0.2612974, -0.3885958, 0.3872799
9: -0.1317314, 0.2451840, -0.1407588, 0.2545699, -0.3863013, 0.3859428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4839497, upper bound: 0.4788160
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4839497, upper bound: 0.4812841
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1812438, 0.1262368, -0.1979244, 0.1397889, -0.3210327, 0.3241612
1: -0.1692075, 0.1270077, -0.1834880, 0.1464162, -0.3156238, 0.3104957
2: -0.1048302, 0.2237196, -0.1223550, 0.2413696, -0.3461998, 0.3460746
3: -0.0883701, 0.2593294, -0.1115895, 0.2932118, -0.3735794, 0.3645959
4: -0.1461221, 0.1663492, -0.1599081, 0.1884172, -0.3345394, 0.3262573
5: -0.1313906, 0.2079459, -0.1488572, 0.2248771, -0.3562677, 0.3568031
6: -0.1693185, 0.1569325, -0.1863988, 0.1771987, -0.3465172, 0.3433313
7: 0.5932113, 1.0695045, 0.5497698, 1.0866816, -0.4934703, 0.5197347
8: -0.1194964, 0.2423275, -0.1363553, 0.2615465, -0.3810429, 0.3786827
9: -0.1237533, 0.2371575, -0.1409385, 0.2546741, -0.3784274, 0.3780959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4818917, upper bound: 0.4772957
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4818917, upper bound: 0.4783912
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1888432, 0.1317277, -0.2004130, 0.1419846, -0.3308277, 0.3321406
1: -0.1756330, 0.1366000, -0.1855939, 0.1490210, -0.3246540, 0.3221940
2: -0.1128298, 0.2319590, -0.1247960, 0.2438776, -0.3567073, 0.3567550
3: -0.0983431, 0.2761314, -0.1135312, 0.2978688, -0.3894066, 0.3832897
4: -0.1522047, 0.1764771, -0.1620215, 0.1916837, -0.3438883, 0.3384987
5: -0.1392468, 0.2158999, -0.1514867, 0.2273217, -0.3665685, 0.3673866
6: -0.1772040, 0.1662185, -0.1889065, 0.1801380, -0.3573420, 0.3551250
7: 0.5710945, 1.0755857, 0.5440536, 1.0879042, -0.5168098, 0.5315322
8: -0.1272983, 0.2509542, -0.1388672, 0.2643468, -0.3916451, 0.3898214
9: -0.1317314, 0.2451840, -0.1433101, 0.2572468, -0.3889782, 0.3884941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4839114, upper bound: 0.4774562
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4839114, upper bound: 0.4798247
time: 2.12 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1797042, 0.1250554, -0.1938129, 0.1358960, -0.3156002, 0.3188683
1: -0.1678785, 0.1250366, -0.1800087, 0.1424643, -0.3103428, 0.3050453
2: -0.1033318, 0.2221264, -0.1178163, 0.2369544, -0.3402862, 0.3399427
3: -0.0890329, 0.2560883, -0.1007997, 0.2881706, -0.3688738, 0.3503636
4: -0.1447527, 0.1643406, -0.1564491, 0.1830170, -0.3277697, 0.3207898
5: -0.1297989, 0.2061402, -0.1446947, 0.2207694, -0.3505684, 0.3508350
6: -0.1675458, 0.1552557, -0.1823366, 0.1721205, -0.3396663, 0.3375922
7: 0.5971586, 1.0717831, 0.5563824, 1.0760221, -0.4788635, 0.5154007
8: -0.1176828, 0.2406801, -0.1323600, 0.2563326, -0.3740155, 0.3730401
9: -0.1223833, 0.2353981, -0.1365970, 0.2502758, -0.3726591, 0.3719951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4803882, upper bound: 0.4852532
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4803882, upper bound: 0.4872231
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1872669, 0.1307593, -0.1957917, 0.1376383, -0.3249053, 0.3265510
1: -0.1745656, 0.1349699, -0.1817190, 0.1446229, -0.3191885, 0.3166889
2: -0.1116759, 0.2305156, -0.1198079, 0.2389458, -0.3506218, 0.3503234
3: -0.0993226, 0.2733757, -0.1027307, 0.2922405, -0.3844348, 0.3696425
4: -0.1510675, 0.1744219, -0.1581436, 0.1856094, -0.3366768, 0.3325655
5: -0.1377511, 0.2143593, -0.1468158, 0.2227110, -0.3604621, 0.3611751
6: -0.1757052, 0.1645198, -0.1843564, 0.1744584, -0.3501636, 0.3488762
7: 0.5746942, 1.0781343, 0.5513976, 1.0772233, -0.5025291, 0.5267367
8: -0.1257447, 0.2494220, -0.1343574, 0.2585509, -0.3842955, 0.3837793
9: -0.1304813, 0.2437558, -0.1385188, 0.2523187, -0.3828000, 0.3822746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4818313, upper bound: 0.4852623
time: 1.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4818313, upper bound: 0.4887890
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1797042, 0.1250554, -0.1956325, 0.1375789, -0.3172831, 0.3206879
1: -0.1678785, 0.1250366, -0.1815816, 0.1443198, -0.3121983, 0.3066182
2: -0.1033318, 0.2221264, -0.1198278, 0.2388553, -0.3421871, 0.3419543
3: -0.0890329, 0.2560883, -0.1056831, 0.2910315, -0.3719378, 0.3554723
4: -0.1447527, 0.1643406, -0.1579999, 0.1853769, -0.3301296, 0.3223406
5: -0.1297989, 0.2061402, -0.1465750, 0.2225713, -0.3523702, 0.3527153
6: -0.1675458, 0.1552557, -0.1841727, 0.1743157, -0.3418615, 0.3394283
7: 0.5971586, 1.0717831, 0.5527422, 1.0800741, -0.4829155, 0.5190408
8: -0.1176828, 0.2406801, -0.1341306, 0.2585621, -0.3762449, 0.3748107
9: -0.1223833, 0.2353981, -0.1384891, 0.2521901, -0.3745735, 0.3738872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4784747, upper bound: 0.4820317
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4784747, upper bound: 0.4839497
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1872669, 0.1307593, -0.1981227, 0.1397733, -0.3270403, 0.3288820
1: -0.1745656, 0.1349699, -0.1836794, 0.1469274, -0.3214930, 0.3186493
2: -0.1116759, 0.2305156, -0.1222568, 0.2413561, -0.3530321, 0.3527724
3: -0.0993226, 0.2733757, -0.1076335, 0.2956828, -0.3880672, 0.3747962
4: -0.1510675, 0.1744219, -0.1601131, 0.1886488, -0.3397162, 0.3345349
5: -0.1377511, 0.2143593, -0.1492041, 0.2250202, -0.3627713, 0.3635634
6: -0.1757052, 0.1645198, -0.1866842, 0.1772477, -0.3529528, 0.3512039
7: 0.5746942, 1.0781343, 0.5470352, 1.0812945, -0.5066004, 0.5310991
8: -0.1257447, 0.2494220, -0.1366474, 0.2613557, -0.3871004, 0.3860694
9: -0.1304813, 0.2437558, -0.1408535, 0.2547590, -0.3852403, 0.3846093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4794660, upper bound: 0.4820387
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4794660, upper bound: 0.4847832
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1812525, 0.1264811, -0.1938129, 0.1358960, -0.3171485, 0.3202940
1: -0.1693382, 0.1271015, -0.1800087, 0.1424643, -0.3118024, 0.3071102
2: -0.1052767, 0.2239556, -0.1178163, 0.2369544, -0.3422310, 0.3417719
3: -0.0941070, 0.2588863, -0.1007997, 0.2881706, -0.3741012, 0.3533982
4: -0.1461892, 0.1664826, -0.1564491, 0.1830170, -0.3292062, 0.3229316
5: -0.1314284, 0.2080816, -0.1446947, 0.2207694, -0.3521978, 0.3527763
6: -0.1693996, 0.1571717, -0.1823366, 0.1721205, -0.3415201, 0.3395083
7: 0.5935581, 1.0759324, 0.5563824, 1.0760221, -0.4824641, 0.5195500
8: -0.1195227, 0.2426780, -0.1323600, 0.2563326, -0.3758553, 0.3750380
9: -0.1241652, 0.2373139, -0.1365970, 0.2502758, -0.3744410, 0.3739109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4776477, upper bound: 0.4836415
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4776477, upper bound: 0.4857183
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1889123, 0.1320076, -0.1957917, 0.1376383, -0.3265507, 0.3277993
1: -0.1758672, 0.1367486, -0.1817190, 0.1446229, -0.3204901, 0.3184676
2: -0.1134427, 0.2322798, -0.1198079, 0.2389458, -0.3523886, 0.3520877
3: -0.1041986, 0.2758722, -0.1027307, 0.2922405, -0.3882824, 0.3722779
4: -0.1523146, 0.1766114, -0.1581436, 0.1856094, -0.3379239, 0.3347550
5: -0.1393435, 0.2160334, -0.1468158, 0.2227110, -0.3620545, 0.3628492
6: -0.1773088, 0.1665508, -0.1843564, 0.1744584, -0.3517672, 0.3509072
7: 0.5711976, 1.0821090, 0.5513976, 1.0772233, -0.5060257, 0.5307114
8: -0.1273114, 0.2514289, -0.1343574, 0.2585509, -0.3858623, 0.3857863
9: -0.1322505, 0.2454108, -0.1385188, 0.2523187, -0.3845691, 0.3839296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4788719, upper bound: 0.4836484
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4788719, upper bound: 0.4866729
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1812525, 0.1264811, -0.1956325, 0.1375789, -0.3188314, 0.3221136
1: -0.1693382, 0.1271015, -0.1815816, 0.1443198, -0.3136579, 0.3086831
2: -0.1052767, 0.2239556, -0.1198278, 0.2388553, -0.3441319, 0.3437835
3: -0.0941070, 0.2588863, -0.1056831, 0.2910315, -0.3769042, 0.3582975
4: -0.1461892, 0.1664826, -0.1579999, 0.1853769, -0.3315660, 0.3244825
5: -0.1314284, 0.2080816, -0.1465750, 0.2225713, -0.3539996, 0.3546566
6: -0.1693996, 0.1571717, -0.1841727, 0.1743157, -0.3437153, 0.3413444
7: 0.5935581, 1.0759324, 0.5527422, 1.0800741, -0.4865160, 0.5231901
8: -0.1195227, 0.2426780, -0.1341306, 0.2585621, -0.3780848, 0.3768086
9: -0.1241652, 0.2373139, -0.1384891, 0.2521901, -0.3763554, 0.3758030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4772957, upper bound: 0.4818917
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4772957, upper bound: 0.4839114
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1889123, 0.1320076, -0.1981227, 0.1397733, -0.3286856, 0.3301303
1: -0.1758672, 0.1367486, -0.1836794, 0.1469274, -0.3227946, 0.3204281
2: -0.1134427, 0.2322798, -0.1222568, 0.2413561, -0.3547989, 0.3545367
3: -0.1041986, 0.2758722, -0.1076335, 0.2956828, -0.3916747, 0.3772316
4: -0.1523146, 0.1766114, -0.1601131, 0.1886488, -0.3409633, 0.3367245
5: -0.1393435, 0.2160334, -0.1492041, 0.2250202, -0.3643637, 0.3652375
6: -0.1773088, 0.1665508, -0.1866842, 0.1772477, -0.3545564, 0.3532350
7: 0.5711976, 1.0821090, 0.5470352, 1.0812945, -0.5100969, 0.5350738
8: -0.1273114, 0.2514289, -0.1366474, 0.2613557, -0.3886671, 0.3880763
9: -0.1322505, 0.2454108, -0.1408535, 0.2547590, -0.3870095, 0.3862643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4783912, upper bound: 0.4818949
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4783912, upper bound: 0.4846592
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1797042, 0.1250554, -0.1953661, 0.1374556, -0.3171598, 0.3204215
1: -0.1678785, 0.1250366, -0.1813436, 0.1438548, -0.3117334, 0.3063802
2: -0.1033318, 0.2221264, -0.1197009, 0.2387497, -0.3420815, 0.3418273
3: -0.0890329, 0.2560883, -0.1065595, 0.2893173, -0.3699062, 0.3558712
4: -0.1447527, 0.1643406, -0.1577384, 0.1850716, -0.3298243, 0.3220791
5: -0.1297989, 0.2061402, -0.1462270, 0.2223307, -0.3521297, 0.3523672
6: -0.1675458, 0.1552557, -0.1838282, 0.1741630, -0.3417087, 0.3390839
7: 0.5971586, 1.0717831, 0.5546491, 1.0825963, -0.4854377, 0.5171340
8: -0.1176828, 0.2406801, -0.1338209, 0.2585112, -0.3761940, 0.3745010
9: -0.1223833, 0.2353981, -0.1383950, 0.2520067, -0.3743900, 0.3737930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4809077, upper bound: 0.4853549
time: 1.29 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4809077, upper bound: 0.4872415
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1872669, 0.1307593, -0.1978467, 0.1396426, -0.3269095, 0.3286059
1: -0.1745656, 0.1349699, -0.1834427, 0.1464607, -0.3210263, 0.3184125
2: -0.1116759, 0.2305156, -0.1221327, 0.2412469, -0.3529229, 0.3526483
3: -0.0993226, 0.2733757, -0.1084894, 0.2939796, -0.3860844, 0.3751808
4: -0.1510675, 0.1744219, -0.1598459, 0.1883272, -0.3393947, 0.3342677
5: -0.1377511, 0.2143593, -0.1488501, 0.2247693, -0.3625203, 0.3632094
6: -0.1757052, 0.1645198, -0.1863315, 0.1770898, -0.3527950, 0.3508512
7: 0.5746942, 1.0781343, 0.5489333, 1.0838127, -0.5091186, 0.5292010
8: -0.1257447, 0.2494220, -0.1363257, 0.2612974, -0.3870421, 0.3857477
9: -0.1304813, 0.2437558, -0.1407588, 0.2545699, -0.3850512, 0.3845146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4820892, upper bound: 0.4853729
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4820892, upper bound: 0.4853729
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1797042, 0.1250554, -0.1979244, 0.1397889, -0.3194931, 0.3229798
1: -0.1678785, 0.1250366, -0.1834880, 0.1464162, -0.3142948, 0.3085245
2: -0.1033318, 0.2221264, -0.1223550, 0.2413696, -0.3447014, 0.3444815
3: -0.0890329, 0.2560883, -0.1115895, 0.2932118, -0.3739426, 0.3611463
4: -0.1447527, 0.1643406, -0.1599081, 0.1884172, -0.3331699, 0.3242488
5: -0.1297989, 0.2061402, -0.1488572, 0.2248771, -0.3546760, 0.3549975
6: -0.1675458, 0.1552557, -0.1863988, 0.1771987, -0.3447445, 0.3416545
7: 0.5971586, 1.0717831, 0.5497698, 1.0866816, -0.4895230, 0.5220133
8: -0.1176828, 0.2406801, -0.1363553, 0.2615465, -0.3792293, 0.3770354
9: -0.1223833, 0.2353981, -0.1409385, 0.2546741, -0.3770574, 0.3763365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4790015, upper bound: 0.4820959
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4790015, upper bound: 0.4840348
time: 1.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1872669, 0.1307593, -0.2004130, 0.1419846, -0.3292515, 0.3311722
1: -0.1745656, 0.1349699, -0.1855939, 0.1490210, -0.3235866, 0.3205638
2: -0.1116759, 0.2305156, -0.1247960, 0.2438776, -0.3555535, 0.3553116
3: -0.0993226, 0.2733757, -0.1135312, 0.2978688, -0.3900942, 0.3804498
4: -0.1510675, 0.1744219, -0.1620215, 0.1916837, -0.3427511, 0.3364434
5: -0.1377511, 0.2143593, -0.1514867, 0.2273217, -0.3650728, 0.3658460
6: -0.1757052, 0.1645198, -0.1889065, 0.1801380, -0.3558432, 0.3534262
7: 0.5746942, 1.0781343, 0.5440536, 1.0879042, -0.5132101, 0.5340807
8: -0.1257447, 0.2494220, -0.1388672, 0.2643468, -0.3900914, 0.3882891
9: -0.1304813, 0.2437558, -0.1433101, 0.2572468, -0.3877281, 0.3870659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4798341, upper bound: 0.4821054
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4798341, upper bound: 0.4848903
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1812525, 0.1264811, -0.1953661, 0.1374556, -0.3187081, 0.3218471
1: -0.1693382, 0.1271015, -0.1813436, 0.1438548, -0.3131930, 0.3084451
2: -0.1052767, 0.2239556, -0.1197009, 0.2387497, -0.3440263, 0.3436565
3: -0.0941070, 0.2588863, -0.1065595, 0.2893173, -0.3751304, 0.3589115
4: -0.1461892, 0.1664826, -0.1577384, 0.1850716, -0.3312607, 0.3242210
5: -0.1314284, 0.2080816, -0.1462270, 0.2223307, -0.3537591, 0.3543085
6: -0.1693996, 0.1571717, -0.1838282, 0.1741630, -0.3435626, 0.3410000
7: 0.5935581, 1.0759324, 0.5546491, 1.0825963, -0.4890382, 0.5212833
8: -0.1195227, 0.2426780, -0.1338209, 0.2585112, -0.3780339, 0.3764989
9: -0.1241652, 0.2373139, -0.1383950, 0.2520067, -0.3761719, 0.3757089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4781966, upper bound: 0.4837016
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4781966, upper bound: 0.4857245
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1889123, 0.1320076, -0.1978467, 0.1396426, -0.3285549, 0.3298543
1: -0.1758672, 0.1367486, -0.1834427, 0.1464607, -0.3223279, 0.3201913
2: -0.1134427, 0.2322798, -0.1221327, 0.2412469, -0.3546897, 0.3544126
3: -0.1041986, 0.2758722, -0.1084894, 0.2939796, -0.3899221, 0.3778152
4: -0.1523146, 0.1766114, -0.1598459, 0.1883272, -0.3406418, 0.3364573
5: -0.1393435, 0.2160334, -0.1488501, 0.2247693, -0.3641128, 0.3648836
6: -0.1773088, 0.1665508, -0.1863315, 0.1770898, -0.3543986, 0.3528823
7: 0.5711976, 1.0821090, 0.5489333, 1.0838127, -0.5126151, 0.5331757
8: -0.1273114, 0.2514289, -0.1363257, 0.2612974, -0.3886088, 0.3877546
9: -0.1322505, 0.2454108, -0.1407588, 0.2545699, -0.3868204, 0.3861696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.64 + 596.59 = 600.22 seconds
