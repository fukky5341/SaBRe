## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00886005


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0005251, 0.0009731, -0.0005251, 0.0009731, -0.0011594, 0.0011594)
1: (-0.0011379, 0.0028129, -0.0011379, 0.0028129, -0.0029502, 0.0029502)
2: (0.0121273, 0.0180442, 0.0121273, 0.0180442, -0.0041689, 0.0041689)
3: (-0.0015077, 0.0029416, -0.0015077, 0.0029416, -0.0030404, 0.0030404)
4: (-0.0057703, -0.0016663, -0.0057703, -0.0016663, -0.0036972, 0.0036972)
5: (0.0064333, 0.0108744, 0.0064333, 0.0108744, -0.0030270, 0.0030270)
6: (0.0076123, 0.0105052, 0.0076123, 0.0105052, -0.0028929, 0.0028929)
7: (-0.0220065, -0.0123654, -0.0220065, -0.0123654, -0.0059726, 0.0059726)
8: (0.9607396, 0.9883626, 0.9607396, 0.9883626, -0.0197066, 0.0197066)
9: (0.0010734, 0.0091918, 0.0010734, 0.0091918, -0.0051729, 0.0051729)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.34 + 1.49 = 2.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0135067, upper bound: 0.0135066

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119106, upper bound: 0.0132021
time: 0.61 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0132021, upper bound: 0.0132021
time: 0.55 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.30 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.30
Output dim: 8, lower bound: -0.0119106, upper bound: 0.0132021
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.30
Output dim: 8, lower bound: -0.0132021, upper bound: 0.0132021

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0004651, 0.0009676, -0.0005231, 0.0009729, -0.0010942, 0.0011464
1: -0.0008570, 0.0028046, -0.0011285, 0.0028126, -0.0027509, 0.0029586
2: 0.0121398, 0.0176234, 0.0121278, 0.0180300, -0.0041791, 0.0038543
3: -0.0014983, 0.0026252, -0.0015073, 0.0029309, -0.0030358, 0.0027845
4: -0.0057616, -0.0019582, -0.0057700, -0.0016762, -0.0037030, 0.0035415
5: 0.0064426, 0.0105586, 0.0064336, 0.0108637, -0.0030206, 0.0027689
6: 0.0076250, 0.0105016, 0.0076128, 0.0105050, -0.0028801, 0.0028889
7: -0.0213209, -0.0123858, -0.0219834, -0.0123662, -0.0052326, 0.0058999
8: 0.9627038, 0.9883043, 0.9608058, 0.9883604, -0.0182627, 0.0197672
9: 0.0010905, 0.0086145, 0.0010740, 0.0091724, -0.0051249, 0.0045669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119106, upper bound: 0.0119106
time: 0.61 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119106, upper bound: 0.0132021
time: 0.64 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0005010, 0.0013268, -0.0005166, 0.0009698, -0.0011253, 0.0015209
1: -0.0010253, 0.0033549, -0.0010980, 0.0028079, -0.0032793, 0.0035539
2: 0.0113156, 0.0178755, 0.0121348, 0.0179844, -0.0050819, 0.0045136
3: -0.0021181, 0.0028148, -0.0015020, 0.0028966, -0.0037260, 0.0032217
4: -0.0063333, -0.0017833, -0.0057651, -0.0017078, -0.0043092, 0.0039818
5: 0.0058240, 0.0107478, 0.0064389, 0.0108295, -0.0037104, 0.0031997
6: 0.0067903, 0.0107351, 0.0076199, 0.0105030, -0.0037128, 0.0031152
7: -0.0217317, -0.0110427, -0.0219090, -0.0123776, -0.0056718, 0.0074389
8: 0.9615270, 0.9921522, 0.9610189, 0.9883276, -0.0214714, 0.0239566
9: -0.0000404, 0.0089604, 0.0010836, 0.0091097, -0.0064190, 0.0050282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0132021, upper bound: 0.0119106
time: 0.62 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0132021, upper bound: 0.0132021
time: 0.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.82 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.82
Output dim: 8, lower bound: -0.0119106, upper bound: 0.0119106
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.82
Output dim: 8, lower bound: -0.0119106, upper bound: 0.0132021
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.82
Output dim: 8, lower bound: -0.0132021, upper bound: 0.0119106
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.82
Output dim: 8, lower bound: -0.0132021, upper bound: 0.0132021

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004651, 0.0009676, -0.0004651, 0.0009676, -0.0010834, 0.0010834
1: -0.0008570, 0.0028046, -0.0008570, 0.0028046, -0.0027635, 0.0027635
2: 0.0121398, 0.0176234, 0.0121398, 0.0176234, -0.0038716, 0.0038716
3: -0.0014983, 0.0026252, -0.0014983, 0.0026252, -0.0027967, 0.0027967
4: -0.0057616, -0.0019582, -0.0057616, -0.0019582, -0.0035516, 0.0035516
5: 0.0064426, 0.0105586, 0.0064426, 0.0105586, -0.0027808, 0.0027808
6: 0.0076250, 0.0105016, 0.0076250, 0.0105016, -0.0028767, 0.0028767
7: -0.0213209, -0.0123858, -0.0213209, -0.0123858, -0.0051855, 0.0051855
8: 0.9627038, 0.9883043, 0.9627038, 0.9883043, -0.0183404, 0.0183404
9: 0.0010905, 0.0086145, 0.0010905, 0.0086145, -0.0045489, 0.0045489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115951, upper bound: 0.0119628
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115718, upper bound: 0.0119628
time: 0.77 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004651, 0.0009676, -0.0005010, 0.0013268, -0.0014606, 0.0011385
1: -0.0008570, 0.0028046, -0.0010253, 0.0033549, -0.0034084, 0.0030594
2: 0.0121398, 0.0176234, 0.0113156, 0.0178755, -0.0042737, 0.0048374
3: -0.0014983, 0.0026252, -0.0021181, 0.0028148, -0.0030884, 0.0035229
4: -0.0057616, -0.0019582, -0.0063333, -0.0017833, -0.0039697, 0.0042214
5: 0.0064426, 0.0105586, 0.0058240, 0.0107478, -0.0030721, 0.0035057
6: 0.0076250, 0.0105016, 0.0067903, 0.0107351, -0.0031101, 0.0037114
7: -0.0213209, -0.0123858, -0.0217317, -0.0110427, -0.0067591, 0.0058526
8: 0.9627038, 0.9883043, 0.9615270, 0.9921522, -0.0228492, 0.0202644
9: 0.0010905, 0.0086145, -0.0000404, 0.0089604, -0.0051207, 0.0058741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115951, upper bound: 0.0128159
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115718, upper bound: 0.0128156
time: 0.75 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0005010, 0.0013268, -0.0004651, 0.0009676, -0.0011385, 0.0014606
1: -0.0010253, 0.0033549, -0.0008570, 0.0028046, -0.0030594, 0.0034084
2: 0.0113156, 0.0178755, 0.0121398, 0.0176234, -0.0048374, 0.0042737
3: -0.0021181, 0.0028148, -0.0014983, 0.0026252, -0.0035229, 0.0030884
4: -0.0063333, -0.0017833, -0.0057616, -0.0019582, -0.0042214, 0.0039697
5: 0.0058240, 0.0107478, 0.0064426, 0.0105586, -0.0035057, 0.0030721
6: 0.0067903, 0.0107351, 0.0076250, 0.0105016, -0.0037114, 0.0031101
7: -0.0217317, -0.0110427, -0.0213209, -0.0123858, -0.0058526, 0.0067591
8: 0.9615270, 0.9921522, 0.9627038, 0.9883043, -0.0202644, 0.0228492
9: -0.0000404, 0.0089604, 0.0010905, 0.0086145, -0.0058741, 0.0051207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128077, upper bound: 0.0115718
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128156, upper bound: 0.0115718
time: 0.73 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0005010, 0.0013268, -0.0005010, 0.0013268, -0.0013985, 0.0013985
1: -0.0010253, 0.0033549, -0.0010253, 0.0033549, -0.0034251, 0.0034251
2: 0.0113156, 0.0178755, 0.0113156, 0.0178755, -0.0047319, 0.0047319
3: -0.0021181, 0.0028148, -0.0021181, 0.0028148, -0.0033859, 0.0033859
4: -0.0063333, -0.0017833, -0.0063333, -0.0017833, -0.0045331, 0.0045331
5: 0.0058240, 0.0107478, 0.0058240, 0.0107478, -0.0033636, 0.0033636
6: 0.0067903, 0.0107351, 0.0067903, 0.0107351, -0.0039448, 0.0039448
7: -0.0217317, -0.0110427, -0.0217317, -0.0110427, -0.0060275, 0.0060275
8: 0.9615270, 0.9921522, 0.9615270, 0.9921522, -0.0224907, 0.0224907
9: -0.0000404, 0.0089604, -0.0000404, 0.0089604, -0.0053278, 0.0053278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128077, upper bound: 0.0115718
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128156, upper bound: 0.0115718
time: 0.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.79 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 8, lower bound: -0.0115951, upper bound: 0.0119628
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 8, lower bound: -0.0115718, upper bound: 0.0119628
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 8, lower bound: -0.0115951, upper bound: 0.0128159
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 8, lower bound: -0.0115718, upper bound: 0.0128156
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 8, lower bound: -0.0128077, upper bound: 0.0115718
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 8, lower bound: -0.0128156, upper bound: 0.0115718
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 8, lower bound: -0.0128077, upper bound: 0.0115718
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 8, lower bound: -0.0128156, upper bound: 0.0115718

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004527, 0.0009621, -0.0004648, 0.0009675, -0.0010680, 0.0010775
1: -0.0007992, 0.0027961, -0.0008555, 0.0028044, -0.0027008, 0.0027534
2: 0.0121525, 0.0175369, 0.0121402, 0.0176211, -0.0038563, 0.0037765
3: -0.0014887, 0.0025601, -0.0014980, 0.0026234, -0.0027851, 0.0027208
4: -0.0057528, -0.0020182, -0.0057614, -0.0019598, -0.0035411, 0.0034897
5: 0.0064521, 0.0104936, 0.0064429, 0.0105568, -0.0027693, 0.0027047
6: 0.0076378, 0.0104980, 0.0076253, 0.0105015, -0.0028637, 0.0028727
7: -0.0211799, -0.0124064, -0.0213172, -0.0123863, -0.0050094, 0.0051601
8: 0.9631079, 0.9882451, 0.9627147, 0.9883027, -0.0179034, 0.0182694
9: 0.0011079, 0.0084958, 0.0010910, 0.0086113, -0.0045275, 0.0044000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119551, upper bound: 0.0119551
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119551, upper bound: 0.0119628
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004383, 0.0010074, -0.0004561, 0.0009641, -0.0010620, 0.0011242
1: -0.0007314, 0.0028656, -0.0008151, 0.0027992, -0.0027523, 0.0028617
2: 0.0120485, 0.0174354, 0.0121480, 0.0175607, -0.0040197, 0.0038341
3: -0.0015669, 0.0024838, -0.0014922, 0.0025780, -0.0029084, 0.0027575
4: -0.0058250, -0.0020886, -0.0057560, -0.0020017, -0.0036486, 0.0035880
5: 0.0063741, 0.0104174, 0.0064487, 0.0105115, -0.0028924, 0.0027406
6: 0.0075324, 0.0105275, 0.0076332, 0.0104993, -0.0029669, 0.0028943
7: -0.0210145, -0.0122369, -0.0212187, -0.0123990, -0.0049753, 0.0054084
8: 0.9635819, 0.9887307, 0.9629967, 0.9882663, -0.0181905, 0.0190269
9: 0.0009652, 0.0083564, 0.0011017, 0.0085284, -0.0047449, 0.0043946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117247, upper bound: 0.0117138
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117138, upper bound: 0.0117138
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004527, 0.0009621, -0.0005007, 0.0013266, -0.0014452, 0.0011326
1: -0.0007992, 0.0027961, -0.0010238, 0.0033547, -0.0033457, 0.0030493
2: 0.0121525, 0.0175369, 0.0113159, 0.0178732, -0.0042586, 0.0047423
3: -0.0014887, 0.0025601, -0.0021178, 0.0028130, -0.0030770, 0.0034471
4: -0.0057528, -0.0020182, -0.0063331, -0.0017849, -0.0039592, 0.0041596
5: 0.0064521, 0.0104936, 0.0058242, 0.0107460, -0.0030608, 0.0034296
6: 0.0076378, 0.0104980, 0.0067906, 0.0107350, -0.0030972, 0.0037075
7: -0.0211799, -0.0124064, -0.0217278, -0.0110432, -0.0065831, 0.0058276
8: 0.9631079, 0.9882451, 0.9615380, 0.9921507, -0.0224121, 0.0201937
9: 0.0011079, 0.0084958, -0.0000400, 0.0089571, -0.0050997, 0.0057252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115717, upper bound: 0.0128023
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115717, upper bound: 0.0128156
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004383, 0.0010074, -0.0004924, 0.0013233, -0.0014392, 0.0011784
1: -0.0007314, 0.0028656, -0.0009847, 0.0033496, -0.0033972, 0.0031522
2: 0.0120485, 0.0174354, 0.0113236, 0.0178147, -0.0044145, 0.0047998
3: -0.0015669, 0.0024838, -0.0021120, 0.0027690, -0.0031920, 0.0034837
4: -0.0058250, -0.0020886, -0.0063278, -0.0018255, -0.0039995, 0.0042392
5: 0.0063741, 0.0104174, 0.0058300, 0.0107021, -0.0031748, 0.0034655
6: 0.0075324, 0.0105275, 0.0067984, 0.0107328, -0.0032004, 0.0037291
7: -0.0210145, -0.0122369, -0.0216325, -0.0110558, -0.0065489, 0.0060629
8: 0.9635819, 0.9887307, 0.9618110, 0.9921147, -0.0226991, 0.0209201
9: 0.0009652, 0.0083564, -0.0000294, 0.0088769, -0.0052986, 0.0057197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113124, upper bound: 0.0125071
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113322, upper bound: 0.0125071
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004883, 0.0013215, -0.0004648, 0.0009675, -0.0011241, 0.0014548
1: -0.0009655, 0.0033468, -0.0008555, 0.0028044, -0.0029949, 0.0033985
2: 0.0113278, 0.0177859, 0.0121402, 0.0176211, -0.0048224, 0.0041797
3: -0.0021089, 0.0027474, -0.0014980, 0.0026234, -0.0035116, 0.0030176
4: -0.0063249, -0.0018455, -0.0057614, -0.0019598, -0.0042113, 0.0039057
5: 0.0058331, 0.0106805, 0.0064429, 0.0105568, -0.0034944, 0.0030013
6: 0.0068026, 0.0107316, 0.0076253, 0.0105015, -0.0036990, 0.0031063
7: -0.0215857, -0.0110626, -0.0213172, -0.0123863, -0.0056872, 0.0067343
8: 0.9619454, 0.9920954, 0.9627147, 0.9883027, -0.0198213, 0.0227797
9: -0.0000237, 0.0088374, 0.0010910, 0.0086113, -0.0058531, 0.0049821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128023, upper bound: 0.0115717
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128023, upper bound: 0.0115718
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004745, 0.0013730, -0.0004561, 0.0009641, -0.0011163, 0.0015051
1: -0.0009011, 0.0034258, -0.0008151, 0.0027992, -0.0030229, 0.0034887
2: 0.0112094, 0.0176895, 0.0121480, 0.0175607, -0.0049588, 0.0042060
3: -0.0021979, 0.0026748, -0.0014922, 0.0025780, -0.0036146, 0.0030264
4: -0.0064070, -0.0019124, -0.0057560, -0.0020017, -0.0043000, 0.0038436
5: 0.0057443, 0.0106081, 0.0064487, 0.0105115, -0.0035973, 0.0030088
6: 0.0066827, 0.0107652, 0.0076332, 0.0104993, -0.0038166, 0.0031320
7: -0.0214285, -0.0108697, -0.0212187, -0.0123990, -0.0056295, 0.0069386
8: 0.9623956, 0.9926478, 0.9629967, 0.9882663, -0.0199602, 0.0234111
9: -0.0001861, 0.0087051, 0.0011017, 0.0085284, -0.0060334, 0.0049443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113322
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113322
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004883, 0.0013215, -0.0005007, 0.0013266, -0.0013834, 0.0013928
1: -0.0009655, 0.0033468, -0.0010238, 0.0033547, -0.0033680, 0.0034153
2: 0.0113278, 0.0177859, 0.0113159, 0.0178732, -0.0047170, 0.0046413
3: -0.0021089, 0.0027474, -0.0021178, 0.0028130, -0.0033747, 0.0033163
4: -0.0063249, -0.0018455, -0.0063331, -0.0017849, -0.0045229, 0.0044734
5: 0.0058331, 0.0106805, 0.0058242, 0.0107460, -0.0033524, 0.0032940
6: 0.0068026, 0.0107316, 0.0067906, 0.0107350, -0.0039324, 0.0039411
7: -0.0215857, -0.0110626, -0.0217278, -0.0110432, -0.0058502, 0.0060025
8: 0.9619454, 0.9920954, 0.9615380, 0.9921507, -0.0220787, 0.0224214
9: -0.0000237, 0.0088374, -0.0000400, 0.0089571, -0.0053065, 0.0051762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128023, upper bound: 0.0115717
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128023, upper bound: 0.0115718
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004745, 0.0013730, -0.0004924, 0.0013233, -0.0013773, 0.0014443
1: -0.0009011, 0.0034258, -0.0009847, 0.0033496, -0.0034303, 0.0035130
2: 0.0112094, 0.0176895, 0.0113236, 0.0178147, -0.0048676, 0.0047112
3: -0.0021979, 0.0026748, -0.0021120, 0.0027690, -0.0034886, 0.0033568
4: -0.0064070, -0.0019124, -0.0063278, -0.0018255, -0.0045814, 0.0044154
5: 0.0057443, 0.0106081, 0.0058300, 0.0107021, -0.0034662, 0.0033335
6: 0.0066827, 0.0107652, 0.0067984, 0.0107328, -0.0040501, 0.0039668
7: -0.0214285, -0.0108697, -0.0216325, -0.0110558, -0.0058156, 0.0062286
8: 0.9623956, 0.9926478, 0.9618110, 0.9921147, -0.0224299, 0.0231244
9: -0.0001861, 0.0087051, -0.0000294, 0.0088769, -0.0055119, 0.0051761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113322
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113322
time: 0.74 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.85 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.85
Output dim: 8, lower bound: -0.0119551, upper bound: 0.0119551
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.85
Output dim: 8, lower bound: -0.0119551, upper bound: 0.0119628
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.85
Output dim: 8, lower bound: -0.0117247, upper bound: 0.0117138
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.85
Output dim: 8, lower bound: -0.0117138, upper bound: 0.0117138
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.85
Output dim: 8, lower bound: -0.0115717, upper bound: 0.0128023
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.85
Output dim: 8, lower bound: -0.0115717, upper bound: 0.0128156
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.85
Output dim: 8, lower bound: -0.0113124, upper bound: 0.0125071
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.85
Output dim: 8, lower bound: -0.0113322, upper bound: 0.0125071
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.85
Output dim: 8, lower bound: -0.0128023, upper bound: 0.0115717
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.85
Output dim: 8, lower bound: -0.0128023, upper bound: 0.0115718
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.85
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113322
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.85
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113322
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.85
Output dim: 8, lower bound: -0.0128023, upper bound: 0.0115717
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.85
Output dim: 8, lower bound: -0.0128023, upper bound: 0.0115718
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.85
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113322
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.85
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113322

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004527, 0.0009621, -0.0004527, 0.0009621, -0.0010626, 0.0010626
1: -0.0007992, 0.0027961, -0.0007992, 0.0027961, -0.0026925, 0.0026925
2: 0.0121525, 0.0175369, 0.0121525, 0.0175369, -0.0037642, 0.0037642
3: -0.0014887, 0.0025601, -0.0014887, 0.0025601, -0.0027116, 0.0027116
4: -0.0057528, -0.0020182, -0.0057528, -0.0020182, -0.0034811, 0.0034811
5: 0.0064521, 0.0104936, 0.0064521, 0.0104936, -0.0026954, 0.0026954
6: 0.0076378, 0.0104980, 0.0076378, 0.0104980, -0.0028603, 0.0028603
7: -0.0211799, -0.0124064, -0.0211799, -0.0124064, -0.0049893, 0.0049893
8: 0.9631079, 0.9882451, 0.9631079, 0.9882451, -0.0178456, 0.0178456
9: 0.0011079, 0.0084958, 0.0011079, 0.0084958, -0.0043831, 0.0043831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117191, upper bound: 0.0117165
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117191, upper bound: 0.0117016
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004527, 0.0009621, -0.0004383, 0.0010074, -0.0011155, 0.0010599
1: -0.0007992, 0.0027961, -0.0007314, 0.0028656, -0.0028229, 0.0027469
2: 0.0121525, 0.0175369, 0.0120485, 0.0174354, -0.0038370, 0.0039594
3: -0.0014887, 0.0025601, -0.0015669, 0.0024838, -0.0027647, 0.0028583
4: -0.0057528, -0.0020182, -0.0058250, -0.0020886, -0.0035576, 0.0036165
5: 0.0064521, 0.0104936, 0.0063741, 0.0104174, -0.0027485, 0.0028419
6: 0.0076378, 0.0104980, 0.0075324, 0.0105275, -0.0028897, 0.0029656
7: -0.0211799, -0.0124064, -0.0210145, -0.0122369, -0.0053073, 0.0049803
8: 0.9631079, 0.9882451, 0.9635819, 0.9887307, -0.0187569, 0.0181900
9: 0.0011079, 0.0084958, 0.0009652, 0.0083564, -0.0044292, 0.0046509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117191, upper bound: 0.0117247
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117191, upper bound: 0.0117138
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004381, 0.0010043, -0.0004555, 0.0009282, -0.0010257, 0.0011204
1: -0.0007308, 0.0028607, -0.0008123, 0.0027441, -0.0026872, 0.0028071
2: 0.0120558, 0.0174344, 0.0122304, 0.0175565, -0.0039549, 0.0037380
3: -0.0015615, 0.0024830, -0.0014302, 0.0025749, -0.0028656, 0.0026859
4: -0.0058199, -0.0020893, -0.0056988, -0.0020046, -0.0035582, 0.0035174
5: 0.0063795, 0.0104167, 0.0065106, 0.0105084, -0.0028502, 0.0026692
6: 0.0075398, 0.0105254, 0.0077166, 0.0104760, -0.0029362, 0.0028088
7: -0.0210129, -0.0122488, -0.0212119, -0.0125333, -0.0048280, 0.0053841
8: 0.9635863, 0.9886967, 0.9630162, 0.9878817, -0.0177409, 0.0187122
9: 0.0009752, 0.0083551, 0.0012147, 0.0085227, -0.0047099, 0.0042689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117134, upper bound: 0.0117134
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117134, upper bound: 0.0117138
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004377, 0.0009859, -0.0004673, 0.0009024, -0.0010072, 0.0011210
1: -0.0007289, 0.0028325, -0.0008674, 0.0027046, -0.0026544, 0.0028178
2: 0.0120980, 0.0174316, 0.0122895, 0.0176390, -0.0039777, 0.0036966
3: -0.0015297, 0.0024810, -0.0013857, 0.0026368, -0.0028902, 0.0026578
4: -0.0057907, -0.0020912, -0.0056578, -0.0019474, -0.0035251, 0.0034675
5: 0.0064112, 0.0104146, 0.0065550, 0.0105702, -0.0028757, 0.0026415
6: 0.0075826, 0.0105135, 0.0077766, 0.0104592, -0.0028767, 0.0027369
7: -0.0210084, -0.0123176, -0.0213462, -0.0126297, -0.0048074, 0.0055220
8: 0.9635993, 0.9884998, 0.9626315, 0.9876053, -0.0175400, 0.0188174
9: 0.0010331, 0.0083513, 0.0012959, 0.0086358, -0.0048286, 0.0042439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117138, upper bound: 0.0117134
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117138, upper bound: 0.0117138
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004527, 0.0009621, -0.0004883, 0.0013215, -0.0014399, 0.0011187
1: -0.0007992, 0.0027961, -0.0009655, 0.0033468, -0.0033376, 0.0029866
2: 0.0121525, 0.0175369, 0.0113278, 0.0177859, -0.0041673, 0.0047303
3: -0.0014887, 0.0025601, -0.0021089, 0.0027474, -0.0030083, 0.0034380
4: -0.0057528, -0.0020182, -0.0063249, -0.0018455, -0.0038971, 0.0041512
5: 0.0064521, 0.0104936, 0.0058331, 0.0106805, -0.0029920, 0.0034206
6: 0.0076378, 0.0104980, 0.0068026, 0.0107316, -0.0030939, 0.0036955
7: -0.0211799, -0.0124064, -0.0215857, -0.0110626, -0.0065635, 0.0056671
8: 0.9631079, 0.9882451, 0.9619454, 0.9920954, -0.0223560, 0.0197636
9: 0.0011079, 0.0084958, -0.0000237, 0.0088374, -0.0049652, 0.0057087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113505, upper bound: 0.0125257
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113505, upper bound: 0.0125071
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004527, 0.0009621, -0.0004745, 0.0013730, -0.0014964, 0.0011104
1: -0.0007992, 0.0027961, -0.0009011, 0.0034258, -0.0034500, 0.0030078
2: 0.0121525, 0.0175369, 0.0112094, 0.0176895, -0.0041907, 0.0048985
3: -0.0014887, 0.0025601, -0.0021979, 0.0026748, -0.0030180, 0.0035645
4: -0.0057528, -0.0020182, -0.0064070, -0.0019124, -0.0038405, 0.0042679
5: 0.0064521, 0.0104936, 0.0057443, 0.0106081, -0.0030002, 0.0035468
6: 0.0076378, 0.0104980, 0.0066827, 0.0107652, -0.0031274, 0.0038153
7: -0.0211799, -0.0124064, -0.0214285, -0.0108697, -0.0068375, 0.0055784
8: 0.9631079, 0.9882451, 0.9623956, 0.9926478, -0.0231412, 0.0198830
9: 0.0011079, 0.0084958, -0.0001861, 0.0087051, -0.0049084, 0.0059394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113505, upper bound: 0.0125257
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113505, upper bound: 0.0125072
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004381, 0.0010043, -0.0004921, 0.0012876, -0.0014021, 0.0011747
1: -0.0007308, 0.0028607, -0.0009832, 0.0032949, -0.0033351, 0.0031030
2: 0.0120558, 0.0174344, 0.0114055, 0.0178125, -0.0043543, 0.0047083
3: -0.0015615, 0.0024830, -0.0020504, 0.0027673, -0.0031550, 0.0034156
4: -0.0058199, -0.0020893, -0.0062709, -0.0018270, -0.0039779, 0.0041817
5: 0.0063795, 0.0104167, 0.0058915, 0.0107005, -0.0031391, 0.0033975
6: 0.0075398, 0.0105254, 0.0068813, 0.0107096, -0.0031698, 0.0036441
7: -0.0210129, -0.0122488, -0.0216290, -0.0111893, -0.0064091, 0.0060407
8: 0.9635863, 0.9886967, 0.9618212, 0.9917323, -0.0222709, 0.0206218
9: 0.0009752, 0.0083551, 0.0000830, 0.0088739, -0.0052720, 0.0056003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113123, upper bound: 0.0125071
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113123, upper bound: 0.0125071
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004377, 0.0009859, -0.0005046, 0.0012623, -0.0013851, 0.0011711
1: -0.0007289, 0.0028325, -0.0010419, 0.0032562, -0.0033013, 0.0031070
2: 0.0120980, 0.0174316, 0.0114635, 0.0179004, -0.0043843, 0.0046654
3: -0.0015297, 0.0024810, -0.0020068, 0.0028334, -0.0031852, 0.0033864
4: -0.0057907, -0.0020912, -0.0062308, -0.0017661, -0.0039502, 0.0041395
5: 0.0064112, 0.0104146, 0.0059350, 0.0107664, -0.0031694, 0.0033688
6: 0.0075826, 0.0105135, 0.0069400, 0.0106932, -0.0031106, 0.0035735
7: -0.0210084, -0.0123176, -0.0217721, -0.0112837, -0.0063862, 0.0061332
8: 0.9635993, 0.9884998, 0.9614111, 0.9914619, -0.0220633, 0.0207356
9: 0.0010331, 0.0083513, 0.0001625, 0.0089945, -0.0053534, 0.0055733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113322, upper bound: 0.0125071
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113322, upper bound: 0.0125071
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004883, 0.0013215, -0.0004527, 0.0009621, -0.0011187, 0.0014399
1: -0.0009655, 0.0033468, -0.0007992, 0.0027961, -0.0029866, 0.0033376
2: 0.0113278, 0.0177859, 0.0121525, 0.0175369, -0.0047303, 0.0041673
3: -0.0021089, 0.0027474, -0.0014887, 0.0025601, -0.0034380, 0.0030083
4: -0.0063249, -0.0018455, -0.0057528, -0.0020182, -0.0041512, 0.0038971
5: 0.0058331, 0.0106805, 0.0064521, 0.0104936, -0.0034206, 0.0029920
6: 0.0068026, 0.0107316, 0.0076378, 0.0104980, -0.0036955, 0.0030939
7: -0.0215857, -0.0110626, -0.0211799, -0.0124064, -0.0056671, 0.0065635
8: 0.9619454, 0.9920954, 0.9631079, 0.9882451, -0.0197636, 0.0223560
9: -0.0000237, 0.0088374, 0.0011079, 0.0084958, -0.0057087, 0.0049652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113124
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113310
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004883, 0.0013215, -0.0004383, 0.0010074, -0.0011716, 0.0014372
1: -0.0009655, 0.0033468, -0.0007314, 0.0028656, -0.0031170, 0.0033920
2: 0.0113278, 0.0177859, 0.0120485, 0.0174354, -0.0048031, 0.0043625
3: -0.0021089, 0.0027474, -0.0015669, 0.0024838, -0.0034912, 0.0031551
4: -0.0063249, -0.0018455, -0.0058250, -0.0020886, -0.0042277, 0.0039795
5: 0.0058331, 0.0106805, 0.0063741, 0.0104174, -0.0034736, 0.0031386
6: 0.0068026, 0.0107316, 0.0075324, 0.0105275, -0.0037249, 0.0031992
7: -0.0215857, -0.0110626, -0.0210145, -0.0122369, -0.0059852, 0.0065546
8: 0.9619454, 0.9920954, 0.9635819, 0.9887307, -0.0206749, 0.0227004
9: -0.0000237, 0.0088374, 0.0009652, 0.0083564, -0.0057548, 0.0052330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113124
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113322
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004745, 0.0013698, -0.0004555, 0.0009282, -0.0010801, 0.0015013
1: -0.0009010, 0.0034209, -0.0008123, 0.0027441, -0.0029595, 0.0034346
2: 0.0112168, 0.0176893, 0.0122304, 0.0175565, -0.0048946, 0.0041127
3: -0.0021923, 0.0026747, -0.0014302, 0.0025749, -0.0035722, 0.0029569
4: -0.0064018, -0.0019125, -0.0056988, -0.0020046, -0.0042100, 0.0037864
5: 0.0057499, 0.0106080, 0.0065106, 0.0105084, -0.0035555, 0.0029394
6: 0.0066903, 0.0107631, 0.0077166, 0.0104760, -0.0037857, 0.0030464
7: -0.0214282, -0.0108818, -0.0212119, -0.0125333, -0.0054842, 0.0069153
8: 0.9623964, 0.9926132, 0.9630162, 0.9878817, -0.0195228, 0.0230991
9: -0.0001759, 0.0087049, 0.0012147, 0.0085227, -0.0059992, 0.0048209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113123
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113322
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004744, 0.0013521, -0.0004673, 0.0009024, -0.0010618, 0.0015017
1: -0.0009005, 0.0033938, -0.0008674, 0.0027046, -0.0029335, 0.0034420
2: 0.0112575, 0.0176886, 0.0122895, 0.0176390, -0.0049126, 0.0040783
3: -0.0021618, 0.0026742, -0.0013857, 0.0026368, -0.0035932, 0.0029342
4: -0.0063737, -0.0019130, -0.0056578, -0.0019474, -0.0041735, 0.0037448
5: 0.0057803, 0.0106074, 0.0065550, 0.0105702, -0.0035774, 0.0029170
6: 0.0067314, 0.0107516, 0.0077766, 0.0104592, -0.0037279, 0.0029750
7: -0.0214270, -0.0109480, -0.0213462, -0.0126297, -0.0054682, 0.0070453
8: 0.9623998, 0.9924237, 0.9626315, 0.9876053, -0.0193545, 0.0231820
9: -0.0001202, 0.0087039, 0.0012959, 0.0086358, -0.0061113, 0.0048014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113123
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113322
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004883, 0.0013215, -0.0004883, 0.0013215, -0.0013783, 0.0013783
1: -0.0009655, 0.0033468, -0.0009655, 0.0033468, -0.0033598, 0.0033598
2: 0.0113278, 0.0177859, 0.0113278, 0.0177859, -0.0046291, 0.0046291
3: -0.0021089, 0.0027474, -0.0021089, 0.0027474, -0.0033072, 0.0033072
4: -0.0063249, -0.0018455, -0.0063249, -0.0018455, -0.0044649, 0.0044649
5: 0.0058331, 0.0106805, 0.0058331, 0.0106805, -0.0032849, 0.0032849
6: 0.0068026, 0.0107316, 0.0068026, 0.0107316, -0.0039291, 0.0039291
7: -0.0215857, -0.0110626, -0.0215857, -0.0110626, -0.0058304, 0.0058304
8: 0.9619454, 0.9920954, 0.9619454, 0.9920954, -0.0220220, 0.0220220
9: -0.0000237, 0.0088374, -0.0000237, 0.0088374, -0.0051595, 0.0051595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113124
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113310
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004883, 0.0013215, -0.0004745, 0.0013730, -0.0014360, 0.0013735
1: -0.0009655, 0.0033468, -0.0009011, 0.0034258, -0.0034820, 0.0034009
2: 0.0113278, 0.0177859, 0.0112094, 0.0176895, -0.0046880, 0.0048121
3: -0.0021089, 0.0027474, -0.0021979, 0.0026748, -0.0033495, 0.0034448
4: -0.0063249, -0.0018455, -0.0064070, -0.0019124, -0.0044125, 0.0045615
5: 0.0058331, 0.0106805, 0.0057443, 0.0106081, -0.0033269, 0.0034222
6: 0.0068026, 0.0107316, 0.0066827, 0.0107652, -0.0039626, 0.0040489
7: -0.0215857, -0.0110626, -0.0214285, -0.0108697, -0.0061286, 0.0058028
8: 0.9619454, 0.9920954, 0.9623956, 0.9926478, -0.0228761, 0.0223007
9: -0.0000237, 0.0088374, -0.0001861, 0.0087051, -0.0052092, 0.0054106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113124
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113322
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004745, 0.0013698, -0.0004921, 0.0012876, -0.0013406, 0.0014407
1: -0.0009010, 0.0034209, -0.0009832, 0.0032949, -0.0033667, 0.0034547
2: 0.0112168, 0.0176893, 0.0114055, 0.0178125, -0.0047976, 0.0046175
3: -0.0021923, 0.0026747, -0.0020504, 0.0027673, -0.0034429, 0.0032871
4: -0.0064018, -0.0019125, -0.0062709, -0.0018270, -0.0045312, 0.0043585
5: 0.0057499, 0.0106080, 0.0058915, 0.0107005, -0.0034211, 0.0032639
6: 0.0066903, 0.0107631, 0.0068813, 0.0107096, -0.0040194, 0.0038817
7: -0.0214282, -0.0108818, -0.0216290, -0.0111893, -0.0056721, 0.0062074
8: 0.9623964, 0.9926132, 0.9618212, 0.9917323, -0.0219904, 0.0227809
9: -0.0001759, 0.0087049, 0.0000830, 0.0088739, -0.0054769, 0.0050533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113123
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113322
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004744, 0.0013521, -0.0005046, 0.0012623, -0.0013238, 0.0014403
1: -0.0009005, 0.0033938, -0.0010419, 0.0032562, -0.0033360, 0.0034500
2: 0.0112575, 0.0176886, 0.0114635, 0.0179004, -0.0048182, 0.0045790
3: -0.0021618, 0.0026742, -0.0020068, 0.0028334, -0.0034696, 0.0032616
4: -0.0063737, -0.0019130, -0.0062308, -0.0017661, -0.0044953, 0.0043178
5: 0.0057803, 0.0106074, 0.0059350, 0.0107664, -0.0034488, 0.0032388
6: 0.0067314, 0.0107516, 0.0069400, 0.0106932, -0.0039618, 0.0038116
7: -0.0214270, -0.0109480, -0.0217721, -0.0112837, -0.0056569, 0.0063408
8: 0.9623998, 0.9924237, 0.9614111, 0.9914619, -0.0218004, 0.0228449
9: -0.0001202, 0.0087039, 0.0001625, 0.0089945, -0.0055882, 0.0050305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113123
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113322
time: 0.76 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.95 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0117191, upper bound: 0.0117165
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0117191, upper bound: 0.0117016
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0117191, upper bound: 0.0117247
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0117191, upper bound: 0.0117138
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0117134, upper bound: 0.0117134
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0117134, upper bound: 0.0117138
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0117138, upper bound: 0.0117134
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0117138, upper bound: 0.0117138
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0113505, upper bound: 0.0125257
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0113505, upper bound: 0.0125071
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0113505, upper bound: 0.0125257
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0113505, upper bound: 0.0125072
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0113123, upper bound: 0.0125071
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0113123, upper bound: 0.0125071
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0113322, upper bound: 0.0125071
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0113322, upper bound: 0.0125071
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113124
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113310
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113124
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113322
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113123
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113322
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113123
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113322
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113124
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113310
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113124
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113322
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113123
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113322
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113123
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113322

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004521, 0.0009263, -0.0004526, 0.0009589, -0.0010589, 0.0010264
1: -0.0007963, 0.0027411, -0.0007985, 0.0027912, -0.0026391, 0.0026277
2: 0.0122348, 0.0175325, 0.0121599, 0.0175359, -0.0036683, 0.0036973
3: -0.0014268, 0.0025568, -0.0014832, 0.0025593, -0.0026399, 0.0026671
4: -0.0056957, -0.0020212, -0.0057477, -0.0020189, -0.0034105, 0.0033895
5: 0.0065139, 0.0104903, 0.0064577, 0.0104928, -0.0026239, 0.0026520
6: 0.0077211, 0.0104747, 0.0076452, 0.0104959, -0.0027748, 0.0028295
7: -0.0211728, -0.0125405, -0.0211782, -0.0124184, -0.0049638, 0.0048418
8: 0.9631283, 0.9878607, 0.9631127, 0.9882107, -0.0175218, 0.0173969
9: 0.0012209, 0.0084898, 0.0011180, 0.0084943, -0.0042575, 0.0043492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117817, upper bound: 0.0117817
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117817, upper bound: 0.0117817
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004639, 0.0009006, -0.0004524, 0.0009411, -0.0010595, 0.0010084
1: -0.0008517, 0.0027019, -0.0007975, 0.0027640, -0.0026520, 0.0025999
2: 0.0122936, 0.0176155, 0.0122006, 0.0175343, -0.0036336, 0.0037327
3: -0.0013826, 0.0026192, -0.0014525, 0.0025582, -0.0026174, 0.0027042
4: -0.0056549, -0.0019637, -0.0057195, -0.0020200, -0.0033658, 0.0033586
5: 0.0065581, 0.0105526, 0.0064883, 0.0104917, -0.0026019, 0.0026901
6: 0.0077807, 0.0104581, 0.0076865, 0.0104844, -0.0027037, 0.0027715
7: -0.0213079, -0.0126364, -0.0211757, -0.0124849, -0.0051059, 0.0048276
8: 0.9627411, 0.9875861, 0.9631199, 0.9880204, -0.0176661, 0.0172295
9: 0.0013016, 0.0086036, 0.0011740, 0.0084922, -0.0042398, 0.0044695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117817, upper bound: 0.0117819
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117817, upper bound: 0.0117819
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004521, 0.0009263, -0.0004381, 0.0010043, -0.0011117, 0.0010237
1: -0.0007963, 0.0027411, -0.0007308, 0.0028607, -0.0027695, 0.0026821
2: 0.0122348, 0.0175325, 0.0120558, 0.0174344, -0.0037413, 0.0038925
3: -0.0014268, 0.0025568, -0.0015615, 0.0024830, -0.0026933, 0.0028139
4: -0.0056957, -0.0020212, -0.0058199, -0.0020893, -0.0034872, 0.0035249
5: 0.0065139, 0.0104903, 0.0063795, 0.0104167, -0.0026773, 0.0027985
6: 0.0077211, 0.0104747, 0.0075398, 0.0105254, -0.0028043, 0.0029349
7: -0.0211728, -0.0125405, -0.0210129, -0.0122488, -0.0052820, 0.0048332
8: 0.9631283, 0.9878607, 0.9635863, 0.9886967, -0.0184335, 0.0177414
9: 0.0012209, 0.0084898, 0.0009752, 0.0083551, -0.0043032, 0.0046171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117187, upper bound: 0.0117134
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117187, upper bound: 0.0117134
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004639, 0.0009006, -0.0004377, 0.0009859, -0.0011128, 0.0010054
1: -0.0008517, 0.0027019, -0.0007289, 0.0028325, -0.0027819, 0.0026507
2: 0.0122936, 0.0176155, 0.0120980, 0.0174316, -0.0037035, 0.0039274
3: -0.0013826, 0.0026192, -0.0015297, 0.0024810, -0.0026682, 0.0028506
4: -0.0056549, -0.0019637, -0.0057907, -0.0020912, -0.0034393, 0.0034936
5: 0.0065581, 0.0105526, 0.0064112, 0.0104146, -0.0026526, 0.0028362
6: 0.0077807, 0.0104581, 0.0075826, 0.0105135, -0.0027328, 0.0028755
7: -0.0213079, -0.0126364, -0.0210084, -0.0123176, -0.0054231, 0.0048148
8: 0.9627411, 0.9875861, 0.9635993, 0.9884998, -0.0185748, 0.0175568
9: 0.0013016, 0.0086036, 0.0010331, 0.0083513, -0.0042790, 0.0047366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117187, upper bound: 0.0117138
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117187, upper bound: 0.0117138
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004370, 0.0009720, -0.0004555, 0.0009282, -0.0010245, 0.0010868
1: -0.0007255, 0.0028113, -0.0008123, 0.0027441, -0.0026332, 0.0027523
2: 0.0121298, 0.0174265, 0.0122304, 0.0175565, -0.0038729, 0.0036717
3: -0.0015058, 0.0024771, -0.0014302, 0.0025749, -0.0028039, 0.0026410
4: -0.0057686, -0.0020947, -0.0056988, -0.0020046, -0.0035013, 0.0034302
5: 0.0064351, 0.0104108, 0.0065106, 0.0105084, -0.0027886, 0.0026249
6: 0.0076148, 0.0105045, 0.0077166, 0.0104760, -0.0028612, 0.0027878
7: -0.0210001, -0.0123694, -0.0212119, -0.0125333, -0.0048053, 0.0052505
8: 0.9636232, 0.9883512, 0.9630162, 0.9878817, -0.0174171, 0.0183292
9: 0.0010767, 0.0083443, 0.0012147, 0.0085227, -0.0045973, 0.0042349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117247, upper bound: 0.0117015
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117247, upper bound: 0.0117015
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004493, 0.0009464, -0.0004555, 0.0009282, -0.0010417, 0.0010673
1: -0.0007832, 0.0027720, -0.0008123, 0.0027441, -0.0026669, 0.0027336
2: 0.0121886, 0.0175129, 0.0122304, 0.0175565, -0.0038448, 0.0037367
3: -0.0014616, 0.0025421, -0.0014302, 0.0025749, -0.0027828, 0.0026992
4: -0.0057278, -0.0020348, -0.0056988, -0.0020046, -0.0034818, 0.0034194
5: 0.0064792, 0.0104756, 0.0065106, 0.0105084, -0.0027675, 0.0026837
6: 0.0076743, 0.0104878, 0.0077166, 0.0104760, -0.0028017, 0.0027712
7: -0.0211408, -0.0124652, -0.0212119, -0.0125333, -0.0049988, 0.0052047
8: 0.9632200, 0.9880767, 0.9630162, 0.9878817, -0.0177086, 0.0181979
9: 0.0011574, 0.0084628, 0.0012147, 0.0085227, -0.0045587, 0.0044037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117247, upper bound: 0.0117016
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117247, upper bound: 0.0117016
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004370, 0.0009720, -0.0004673, 0.0009024, -0.0010014, 0.0011041
1: -0.0007255, 0.0028113, -0.0008674, 0.0027046, -0.0026166, 0.0027864
2: 0.0121298, 0.0174265, 0.0122895, 0.0176390, -0.0039307, 0.0036469
3: -0.0015058, 0.0024771, -0.0013857, 0.0026368, -0.0028549, 0.0026224
4: -0.0057686, -0.0020947, -0.0056578, -0.0019474, -0.0034925, 0.0034130
5: 0.0064351, 0.0104108, 0.0065550, 0.0105702, -0.0028405, 0.0026063
6: 0.0076148, 0.0105045, 0.0077766, 0.0104592, -0.0028445, 0.0027279
7: -0.0210001, -0.0123694, -0.0213462, -0.0126297, -0.0047649, 0.0054455
8: 0.9636232, 0.9883512, 0.9626315, 0.9876053, -0.0173014, 0.0185982
9: 0.0010767, 0.0083443, 0.0012959, 0.0086358, -0.0047641, 0.0042009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117134, upper bound: 0.0117015
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117134, upper bound: 0.0117015
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004493, 0.0009464, -0.0004673, 0.0009024, -0.0010104, 0.0010749
1: -0.0007832, 0.0027720, -0.0008674, 0.0027046, -0.0026011, 0.0027226
2: 0.0121886, 0.0175129, 0.0122895, 0.0176390, -0.0038422, 0.0036354
3: -0.0014616, 0.0025421, -0.0013857, 0.0026368, -0.0027865, 0.0026195
4: -0.0057278, -0.0020348, -0.0056578, -0.0019474, -0.0034463, 0.0033724
5: 0.0064792, 0.0104756, 0.0065550, 0.0105702, -0.0027720, 0.0026040
6: 0.0076743, 0.0104878, 0.0077766, 0.0104592, -0.0027849, 0.0027113
7: -0.0211408, -0.0124652, -0.0213462, -0.0126297, -0.0048308, 0.0052790
8: 0.9632200, 0.9880767, 0.9626315, 0.9876053, -0.0172368, 0.0181780
9: 0.0011574, 0.0084628, 0.0012959, 0.0086358, -0.0046054, 0.0042401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117134, upper bound: 0.0117016
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117134, upper bound: 0.0117016
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004521, 0.0009263, -0.0004882, 0.0013183, -0.0014362, 0.0010826
1: -0.0007963, 0.0027411, -0.0009654, 0.0033419, -0.0032846, 0.0029235
2: 0.0122348, 0.0175325, 0.0113351, 0.0177857, -0.0040741, 0.0046639
3: -0.0014268, 0.0025568, -0.0021033, 0.0027472, -0.0029390, 0.0033939
4: -0.0056957, -0.0020212, -0.0063198, -0.0018456, -0.0038283, 0.0040600
5: 0.0065139, 0.0104903, 0.0058387, 0.0106804, -0.0029230, 0.0033775
6: 0.0077211, 0.0104747, 0.0068101, 0.0107295, -0.0030084, 0.0036647
7: -0.0211728, -0.0125405, -0.0215853, -0.0110746, -0.0065389, 0.0055219
8: 0.9631283, 0.9878607, 0.9619464, 0.9920610, -0.0220348, 0.0193274
9: 0.0012209, 0.0084898, -0.0000136, 0.0088372, -0.0048421, 0.0056756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113937, upper bound: 0.0126390
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113937, upper bound: 0.0126390
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004639, 0.0009006, -0.0004881, 0.0013006, -0.0014371, 0.0010647
1: -0.0008517, 0.0027019, -0.0009648, 0.0033148, -0.0032952, 0.0028981
2: 0.0122936, 0.0176155, 0.0113757, 0.0177849, -0.0040421, 0.0046961
3: -0.0013826, 0.0026192, -0.0020729, 0.0027466, -0.0029193, 0.0034286
4: -0.0056549, -0.0019637, -0.0062917, -0.0018462, -0.0037887, 0.0040268
5: 0.0065581, 0.0105526, 0.0058691, 0.0106798, -0.0029041, 0.0034132
6: 0.0077807, 0.0104581, 0.0068511, 0.0107181, -0.0029374, 0.0036070
7: -0.0213079, -0.0126364, -0.0215840, -0.0111406, -0.0066756, 0.0055085
8: 0.9627411, 0.9875861, 0.9619502, 0.9918718, -0.0221635, 0.0191719
9: 0.0013016, 0.0086036, 0.0000420, 0.0088360, -0.0048263, 0.0057913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113937, upper bound: 0.0126390
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113937, upper bound: 0.0126391
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004521, 0.0009263, -0.0004745, 0.0013698, -0.0014926, 0.0010743
1: -0.0007963, 0.0027411, -0.0009010, 0.0034209, -0.0033970, 0.0029447
2: 0.0122348, 0.0175325, 0.0112168, 0.0176893, -0.0040973, 0.0048322
3: -0.0014268, 0.0025568, -0.0021923, 0.0026747, -0.0029483, 0.0035205
4: -0.0056957, -0.0020212, -0.0064018, -0.0019125, -0.0037833, 0.0041767
5: 0.0065139, 0.0104903, 0.0057499, 0.0106080, -0.0029307, 0.0035038
6: 0.0077211, 0.0104747, 0.0066903, 0.0107631, -0.0030419, 0.0037845
7: -0.0211728, -0.0125405, -0.0214282, -0.0108818, -0.0068131, 0.0054333
8: 0.9631283, 0.9878607, 0.9623964, 0.9926132, -0.0228204, 0.0194454
9: 0.0012209, 0.0084898, -0.0001759, 0.0087049, -0.0047852, 0.0059065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113338, upper bound: 0.0125072
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113338, upper bound: 0.0125072
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004639, 0.0009006, -0.0004744, 0.0013521, -0.0014936, 0.0010564
1: -0.0008517, 0.0027019, -0.0009005, 0.0033938, -0.0034062, 0.0029194
2: 0.0122936, 0.0176155, 0.0112575, 0.0176886, -0.0040658, 0.0048623
3: -0.0013826, 0.0026192, -0.0021618, 0.0026742, -0.0029282, 0.0035536
4: -0.0056549, -0.0019637, -0.0063737, -0.0019130, -0.0037419, 0.0041421
5: 0.0065581, 0.0105526, 0.0057803, 0.0106074, -0.0029111, 0.0035380
6: 0.0077807, 0.0104581, 0.0067314, 0.0107516, -0.0029708, 0.0037267
7: -0.0213079, -0.0126364, -0.0214270, -0.0109480, -0.0069464, 0.0054201
8: 0.9627411, 0.9875861, 0.9623998, 0.9924237, -0.0229394, 0.0192907
9: 0.0013016, 0.0086036, -0.0001202, 0.0087039, -0.0047684, 0.0060193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113338, upper bound: 0.0125072
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113338, upper bound: 0.0125072
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004370, 0.0009720, -0.0004921, 0.0012876, -0.0014008, 0.0011411
1: -0.0007255, 0.0028113, -0.0009832, 0.0032949, -0.0032811, 0.0030482
2: 0.0121298, 0.0174265, 0.0114055, 0.0178125, -0.0042723, 0.0046420
3: -0.0015058, 0.0024771, -0.0020504, 0.0027673, -0.0030933, 0.0033707
4: -0.0057686, -0.0020947, -0.0062709, -0.0018270, -0.0039210, 0.0041032
5: 0.0064351, 0.0104108, 0.0058915, 0.0107005, -0.0030775, 0.0033532
6: 0.0076148, 0.0105045, 0.0068813, 0.0107096, -0.0030948, 0.0036231
7: -0.0210001, -0.0123694, -0.0216290, -0.0111893, -0.0063864, 0.0059070
8: 0.9636232, 0.9883512, 0.9618212, 0.9917323, -0.0219471, 0.0202388
9: 0.0010767, 0.0083443, 0.0000830, 0.0088739, -0.0051594, 0.0055663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113124, upper bound: 0.0125004
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113124, upper bound: 0.0125004
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004493, 0.0009464, -0.0004921, 0.0012876, -0.0014181, 0.0011215
1: -0.0007832, 0.0027720, -0.0009832, 0.0032949, -0.0033148, 0.0030294
2: 0.0121886, 0.0175129, 0.0114055, 0.0178125, -0.0042442, 0.0047071
3: -0.0014616, 0.0025421, -0.0020504, 0.0027673, -0.0030722, 0.0034289
4: -0.0057278, -0.0020348, -0.0062709, -0.0018270, -0.0039008, 0.0040924
5: 0.0064792, 0.0104756, 0.0058915, 0.0107005, -0.0030564, 0.0034121
6: 0.0076743, 0.0104878, 0.0068813, 0.0107096, -0.0030353, 0.0036065
7: -0.0211408, -0.0124652, -0.0216290, -0.0111893, -0.0065799, 0.0058612
8: 0.9632200, 0.9880767, 0.9618212, 0.9917323, -0.0222386, 0.0201076
9: 0.0011574, 0.0084628, 0.0000830, 0.0088739, -0.0051208, 0.0057351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113124, upper bound: 0.0125004
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113124, upper bound: 0.0125004
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004370, 0.0009720, -0.0005046, 0.0012623, -0.0013791, 0.0011542
1: -0.0007255, 0.0028113, -0.0010419, 0.0032562, -0.0032500, 0.0030756
2: 0.0121298, 0.0174265, 0.0114635, 0.0179004, -0.0043373, 0.0045955
3: -0.0015058, 0.0024771, -0.0020068, 0.0028334, -0.0031499, 0.0033357
4: -0.0057686, -0.0020947, -0.0062308, -0.0017661, -0.0039176, 0.0040710
5: 0.0064351, 0.0104108, 0.0059350, 0.0107664, -0.0031341, 0.0033183
6: 0.0076148, 0.0105045, 0.0069400, 0.0106932, -0.0030784, 0.0035645
7: -0.0210001, -0.0123694, -0.0217721, -0.0112837, -0.0063105, 0.0060567
8: 0.9636232, 0.9883512, 0.9614111, 0.9914619, -0.0217298, 0.0205164
9: 0.0010767, 0.0083443, 0.0001625, 0.0089945, -0.0052890, 0.0055024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113123, upper bound: 0.0125004
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113123, upper bound: 0.0125004
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004493, 0.0009464, -0.0005046, 0.0012623, -0.0013883, 0.0011283
1: -0.0007832, 0.0027720, -0.0010419, 0.0032562, -0.0032481, 0.0030271
2: 0.0121886, 0.0175129, 0.0114635, 0.0179004, -0.0042524, 0.0046043
3: -0.0014616, 0.0025421, -0.0020068, 0.0028334, -0.0030847, 0.0033481
4: -0.0057278, -0.0020348, -0.0062308, -0.0017661, -0.0038842, 0.0040444
5: 0.0064792, 0.0104756, 0.0059350, 0.0107664, -0.0030695, 0.0033312
6: 0.0076743, 0.0104878, 0.0069400, 0.0106932, -0.0030189, 0.0035478
7: -0.0211408, -0.0124652, -0.0217721, -0.0112837, -0.0064095, 0.0059317
8: 0.9632200, 0.9880767, 0.9614111, 0.9914619, -0.0217601, 0.0201287
9: 0.0011574, 0.0084628, 0.0001625, 0.0089945, -0.0051759, 0.0055695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113123, upper bound: 0.0125004
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113123, upper bound: 0.0125004
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004879, 0.0012858, -0.0004526, 0.0009589, -0.0011151, 0.0014028
1: -0.0009639, 0.0032921, -0.0007985, 0.0027912, -0.0029375, 0.0032758
2: 0.0114097, 0.0177836, 0.0121599, 0.0175359, -0.0046390, 0.0041086
3: -0.0020473, 0.0027456, -0.0014832, 0.0025593, -0.0033699, 0.0029731
4: -0.0062681, -0.0018471, -0.0057477, -0.0020189, -0.0040838, 0.0038103
5: 0.0058946, 0.0106788, 0.0064577, 0.0104928, -0.0033525, 0.0029579
6: 0.0068855, 0.0107084, 0.0076452, 0.0104959, -0.0036104, 0.0030632
7: -0.0215819, -0.0111960, -0.0211782, -0.0124184, -0.0056443, 0.0064235
8: 0.9619563, 0.9917130, 0.9631127, 0.9882107, -0.0194746, 0.0219287
9: 0.0000887, 0.0088343, 0.0011180, 0.0084943, -0.0055894, 0.0049375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126390, upper bound: 0.0113938
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126390, upper bound: 0.0113938
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0005002, 0.0012606, -0.0004524, 0.0009411, -0.0011117, 0.0013864
1: -0.0010216, 0.0032535, -0.0007975, 0.0027640, -0.0029465, 0.0032470
2: 0.0114675, 0.0178700, 0.0122006, 0.0175343, -0.0046028, 0.0041499
3: -0.0020039, 0.0028106, -0.0014525, 0.0025582, -0.0033462, 0.0030098
4: -0.0062280, -0.0017872, -0.0057195, -0.0020200, -0.0040380, 0.0037857
5: 0.0059380, 0.0107436, 0.0064883, 0.0104917, -0.0033294, 0.0029944
6: 0.0069440, 0.0106921, 0.0076865, 0.0104844, -0.0035404, 0.0030055
7: -0.0217226, -0.0112902, -0.0211757, -0.0124849, -0.0057394, 0.0064068
8: 0.9615530, 0.9914432, 0.9631199, 0.9880204, -0.0196344, 0.0217542
9: 0.0001680, 0.0089527, 0.0011740, 0.0084922, -0.0055696, 0.0050218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126390, upper bound: 0.0114116
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126390, upper bound: 0.0114116
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004879, 0.0012858, -0.0004381, 0.0010043, -0.0011678, 0.0014001
1: -0.0009639, 0.0032921, -0.0007308, 0.0028607, -0.0030679, 0.0033303
2: 0.0114097, 0.0177836, 0.0120558, 0.0174344, -0.0047120, 0.0043039
3: -0.0020473, 0.0027456, -0.0015615, 0.0024830, -0.0034232, 0.0031199
4: -0.0062681, -0.0018471, -0.0058199, -0.0020893, -0.0041605, 0.0039457
5: 0.0058946, 0.0106788, 0.0063795, 0.0104167, -0.0034059, 0.0031045
6: 0.0068855, 0.0107084, 0.0075398, 0.0105254, -0.0036399, 0.0031686
7: -0.0215819, -0.0111960, -0.0210129, -0.0122488, -0.0059625, 0.0064150
8: 0.9619563, 0.9917130, 0.9635863, 0.9886967, -0.0203863, 0.0222733
9: 0.0000887, 0.0088343, 0.0009752, 0.0083551, -0.0056351, 0.0052054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113123
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113123
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0005002, 0.0012606, -0.0004377, 0.0009859, -0.0011650, 0.0013834
1: -0.0010216, 0.0032535, -0.0007289, 0.0028325, -0.0030765, 0.0032979
2: 0.0114675, 0.0178700, 0.0120980, 0.0174316, -0.0046727, 0.0043446
3: -0.0020039, 0.0028106, -0.0015297, 0.0024810, -0.0033970, 0.0031562
4: -0.0062280, -0.0017872, -0.0057907, -0.0020912, -0.0041115, 0.0039207
5: 0.0059380, 0.0107436, 0.0064112, 0.0104146, -0.0033800, 0.0031405
6: 0.0069440, 0.0106921, 0.0075826, 0.0105135, -0.0035694, 0.0031095
7: -0.0217226, -0.0112902, -0.0210084, -0.0123176, -0.0060566, 0.0063940
8: 0.9615530, 0.9914432, 0.9635993, 0.9884998, -0.0205432, 0.0220816
9: 0.0001680, 0.0089527, 0.0010331, 0.0083513, -0.0056089, 0.0052889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113322
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113322
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004742, 0.0013369, -0.0004555, 0.0009282, -0.0010797, 0.0014675
1: -0.0008998, 0.0033706, -0.0008123, 0.0027441, -0.0029186, 0.0033841
2: 0.0112922, 0.0176875, 0.0122304, 0.0175565, -0.0048189, 0.0040653
3: -0.0021356, 0.0026734, -0.0014302, 0.0025749, -0.0035153, 0.0029282
4: -0.0063495, -0.0019137, -0.0056988, -0.0020046, -0.0041575, 0.0037851
5: 0.0058064, 0.0106067, 0.0065106, 0.0105084, -0.0034987, 0.0029115
6: 0.0067666, 0.0107417, 0.0077166, 0.0104760, -0.0037094, 0.0030251
7: -0.0214253, -0.0110046, -0.0212119, -0.0125333, -0.0054760, 0.0067920
8: 0.9624048, 0.9922614, 0.9630162, 0.9878817, -0.0192871, 0.0227458
9: -0.0000725, 0.0087024, 0.0012147, 0.0085227, -0.0058954, 0.0048036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113143
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113142
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004871, 0.0013133, -0.0004555, 0.0009282, -0.0010932, 0.0014469
1: -0.0009600, 0.0033343, -0.0008123, 0.0027441, -0.0029492, 0.0033491
2: 0.0113465, 0.0177776, 0.0122304, 0.0175565, -0.0047667, 0.0041370
3: -0.0020948, 0.0027411, -0.0014302, 0.0025749, -0.0034760, 0.0029928
4: -0.0063119, -0.0018512, -0.0056988, -0.0020046, -0.0041212, 0.0038410
5: 0.0058471, 0.0106743, 0.0065106, 0.0105084, -0.0034594, 0.0029764
6: 0.0068215, 0.0107263, 0.0077166, 0.0104760, -0.0036545, 0.0030097
7: -0.0215721, -0.0110930, -0.0212119, -0.0125333, -0.0056265, 0.0067068
8: 0.9619842, 0.9920081, 0.9630162, 0.9878817, -0.0195969, 0.0225018
9: 0.0000019, 0.0088260, 0.0012147, 0.0085227, -0.0058237, 0.0049366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113310
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113310
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004742, 0.0013369, -0.0004673, 0.0009024, -0.0010567, 0.0014847
1: -0.0008998, 0.0033706, -0.0008674, 0.0027046, -0.0029021, 0.0034181
2: 0.0112922, 0.0176875, 0.0122895, 0.0176390, -0.0048768, 0.0040405
3: -0.0021356, 0.0026734, -0.0013857, 0.0026368, -0.0035663, 0.0029096
4: -0.0063495, -0.0019137, -0.0056578, -0.0019474, -0.0041487, 0.0037441
5: 0.0058064, 0.0106067, 0.0065550, 0.0105702, -0.0035506, 0.0028929
6: 0.0067666, 0.0107417, 0.0077766, 0.0104592, -0.0036926, 0.0029651
7: -0.0214253, -0.0110046, -0.0213462, -0.0126297, -0.0054357, 0.0069870
8: 0.9624048, 0.9922614, 0.9626315, 0.9876053, -0.0191714, 0.0230149
9: -0.0000725, 0.0087024, 0.0012959, 0.0086358, -0.0060622, 0.0047696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113123
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113123
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004871, 0.0013133, -0.0004673, 0.0009024, -0.0010649, 0.0014564
1: -0.0009600, 0.0033343, -0.0008674, 0.0027046, -0.0028987, 0.0033532
2: 0.0113465, 0.0177776, 0.0122895, 0.0176390, -0.0047867, 0.0040400
3: -0.0020948, 0.0027411, -0.0013857, 0.0026368, -0.0034967, 0.0029148
4: -0.0063119, -0.0018512, -0.0056578, -0.0019474, -0.0041014, 0.0038011
5: 0.0058471, 0.0106743, 0.0065550, 0.0105702, -0.0034810, 0.0028995
6: 0.0068215, 0.0107263, 0.0077766, 0.0104592, -0.0036377, 0.0029498
7: -0.0215721, -0.0110930, -0.0213462, -0.0126297, -0.0054983, 0.0068179
8: 0.9619842, 0.9920081, 0.9626315, 0.9876053, -0.0191622, 0.0225873
9: 0.0000019, 0.0088260, 0.0012959, 0.0086358, -0.0059013, 0.0048171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113310
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113310
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004879, 0.0012858, -0.0004882, 0.0013183, -0.0013747, 0.0013415
1: -0.0009639, 0.0032921, -0.0009654, 0.0033419, -0.0033001, 0.0032965
2: 0.0114097, 0.0177836, 0.0113351, 0.0177857, -0.0045358, 0.0045572
3: -0.0020473, 0.0027456, -0.0021033, 0.0027472, -0.0032376, 0.0032614
4: -0.0062681, -0.0018471, -0.0063198, -0.0018456, -0.0043964, 0.0043741
5: 0.0058946, 0.0106788, 0.0058387, 0.0106804, -0.0032156, 0.0032401
6: 0.0068855, 0.0107084, 0.0068101, 0.0107295, -0.0038440, 0.0038984
7: -0.0215819, -0.0111960, -0.0215853, -0.0110746, -0.0058095, 0.0056871
8: 0.9619563, 0.9917130, 0.9619464, 0.9920610, -0.0216675, 0.0215844
9: 0.0000887, 0.0088343, -0.0000136, 0.0088372, -0.0050375, 0.0051274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126390, upper bound: 0.0113938
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126390, upper bound: 0.0113938
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0005002, 0.0012606, -0.0004881, 0.0013006, -0.0013747, 0.0013250
1: -0.0010216, 0.0032535, -0.0009648, 0.0033148, -0.0032964, 0.0032676
2: 0.0114675, 0.0178700, 0.0113757, 0.0177849, -0.0044999, 0.0045816
3: -0.0020039, 0.0028106, -0.0020729, 0.0027466, -0.0032142, 0.0032901
4: -0.0062280, -0.0017872, -0.0062917, -0.0018462, -0.0043519, 0.0043353
5: 0.0059380, 0.0107436, 0.0058691, 0.0106798, -0.0031924, 0.0032695
6: 0.0069440, 0.0106921, 0.0068511, 0.0107181, -0.0037740, 0.0038410
7: -0.0217226, -0.0112902, -0.0215840, -0.0111406, -0.0059425, 0.0056748
8: 0.9615530, 0.9914432, 0.9619502, 0.9918718, -0.0217447, 0.0214086
9: 0.0001680, 0.0089527, 0.0000420, 0.0088360, -0.0050198, 0.0052453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126390, upper bound: 0.0114116
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126390, upper bound: 0.0114116
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004879, 0.0012858, -0.0004745, 0.0013698, -0.0014324, 0.0013367
1: -0.0009639, 0.0032921, -0.0009010, 0.0034209, -0.0034222, 0.0033376
2: 0.0114097, 0.0177836, 0.0112168, 0.0176893, -0.0045948, 0.0047400
3: -0.0020473, 0.0027456, -0.0021923, 0.0026747, -0.0032800, 0.0033988
4: -0.0062681, -0.0018471, -0.0064018, -0.0019125, -0.0043556, 0.0045008
5: 0.0058946, 0.0106788, 0.0057499, 0.0106080, -0.0032575, 0.0033773
6: 0.0068855, 0.0107084, 0.0066903, 0.0107631, -0.0038775, 0.0040182
7: -0.0215819, -0.0111960, -0.0214282, -0.0108818, -0.0061073, 0.0056594
8: 0.9619563, 0.9917130, 0.9623964, 0.9926132, -0.0225208, 0.0218638
9: 0.0000887, 0.0088343, -0.0001759, 0.0087049, -0.0050865, 0.0053781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113123
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113123
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0005002, 0.0012606, -0.0004744, 0.0013521, -0.0014326, 0.0013202
1: -0.0010216, 0.0032535, -0.0009005, 0.0033938, -0.0034189, 0.0033089
2: 0.0114675, 0.0178700, 0.0112575, 0.0176886, -0.0045594, 0.0047651
3: -0.0020039, 0.0028106, -0.0021618, 0.0026742, -0.0032565, 0.0034281
4: -0.0062280, -0.0017872, -0.0063737, -0.0019130, -0.0043150, 0.0044625
5: 0.0059380, 0.0107436, 0.0057803, 0.0106074, -0.0032344, 0.0034072
6: 0.0069440, 0.0106921, 0.0067314, 0.0107516, -0.0038075, 0.0039607
7: -0.0217226, -0.0112902, -0.0214270, -0.0109480, -0.0062415, 0.0056465
8: 0.9615530, 0.9914432, 0.9623998, 0.9924237, -0.0226013, 0.0216884
9: 0.0001680, 0.0089527, -0.0001202, 0.0087039, -0.0050662, 0.0054970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113322
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113322
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004742, 0.0013369, -0.0004921, 0.0012876, -0.0013404, 0.0014068
1: -0.0008998, 0.0033706, -0.0009832, 0.0032949, -0.0033160, 0.0034007
2: 0.0112922, 0.0176875, 0.0114055, 0.0178125, -0.0047166, 0.0045570
3: -0.0021356, 0.0026734, -0.0020504, 0.0027673, -0.0033820, 0.0032489
4: -0.0063495, -0.0019137, -0.0062709, -0.0018270, -0.0044751, 0.0043572
5: 0.0058064, 0.0106067, 0.0058915, 0.0107005, -0.0033603, 0.0032266
6: 0.0067666, 0.0107417, 0.0068813, 0.0107096, -0.0039430, 0.0038604
7: -0.0214253, -0.0110046, -0.0216290, -0.0111893, -0.0056652, 0.0060754
8: 0.9624048, 0.9922614, 0.9618212, 0.9917323, -0.0216883, 0.0224028
9: -0.0000725, 0.0087024, 0.0000830, 0.0088739, -0.0053658, 0.0050293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113143
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113143
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004871, 0.0013133, -0.0004921, 0.0012876, -0.0013568, 0.0013870
1: -0.0009600, 0.0033343, -0.0009832, 0.0032949, -0.0033351, 0.0033792
2: 0.0113465, 0.0177776, 0.0114055, 0.0178125, -0.0046844, 0.0046095
3: -0.0020948, 0.0027411, -0.0020504, 0.0027673, -0.0033578, 0.0033035
4: -0.0063119, -0.0018512, -0.0062709, -0.0018270, -0.0044528, 0.0044191
5: 0.0058471, 0.0106743, 0.0058915, 0.0107005, -0.0033362, 0.0032821
6: 0.0068215, 0.0107263, 0.0068813, 0.0107096, -0.0038881, 0.0038450
7: -0.0215721, -0.0110930, -0.0216290, -0.0111893, -0.0058546, 0.0060230
8: 0.9619842, 0.9920081, 0.9618212, 0.9917323, -0.0219066, 0.0222527
9: 0.0000019, 0.0088260, 0.0000830, 0.0088739, -0.0053217, 0.0051967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113310
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113310
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004742, 0.0013369, -0.0005046, 0.0012623, -0.0013189, 0.0014230
1: -0.0008998, 0.0033706, -0.0010419, 0.0032562, -0.0032917, 0.0034196
2: 0.0112922, 0.0176875, 0.0114635, 0.0179004, -0.0047727, 0.0045205
3: -0.0021356, 0.0026734, -0.0020068, 0.0028334, -0.0034354, 0.0032215
4: -0.0063495, -0.0019137, -0.0062308, -0.0017661, -0.0044638, 0.0043170
5: 0.0058064, 0.0106067, 0.0059350, 0.0107664, -0.0034146, 0.0031992
6: 0.0067666, 0.0107417, 0.0069400, 0.0106932, -0.0039266, 0.0038017
7: -0.0214253, -0.0110046, -0.0217721, -0.0112837, -0.0056058, 0.0062666
8: 0.9624048, 0.9922614, 0.9614111, 0.9914619, -0.0215182, 0.0226325
9: -0.0000725, 0.0087024, 0.0001625, 0.0089945, -0.0055258, 0.0049793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113123
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113123
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004871, 0.0013133, -0.0005046, 0.0012623, -0.0013276, 0.0013959
1: -0.0009600, 0.0033343, -0.0010419, 0.0032562, -0.0032830, 0.0033702
2: 0.0113465, 0.0177776, 0.0114635, 0.0179004, -0.0046801, 0.0045182
3: -0.0020948, 0.0027411, -0.0020068, 0.0028334, -0.0033611, 0.0032257
4: -0.0063119, -0.0018512, -0.0062308, -0.0017661, -0.0044170, 0.0043771
5: 0.0058471, 0.0106743, 0.0059350, 0.0107664, -0.0033405, 0.0032040
6: 0.0068215, 0.0107263, 0.0069400, 0.0106932, -0.0038717, 0.0037863
7: -0.0215721, -0.0110930, -0.0217721, -0.0112837, -0.0056922, 0.0061081
8: 0.9619842, 0.9920081, 0.9614111, 0.9914619, -0.0214989, 0.0222221
9: 0.0000019, 0.0088260, 0.0001625, 0.0089945, -0.0053742, 0.0050333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113310
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113310
time: 0.75 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.97 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0117817, upper bound: 0.0117817
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0117817, upper bound: 0.0117817
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0117817, upper bound: 0.0117819
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0117817, upper bound: 0.0117819
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0117187, upper bound: 0.0117134
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0117187, upper bound: 0.0117134
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0117187, upper bound: 0.0117138
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0117187, upper bound: 0.0117138
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0117247, upper bound: 0.0117015
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0117247, upper bound: 0.0117015
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0117247, upper bound: 0.0117016
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0117247, upper bound: 0.0117016
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0117134, upper bound: 0.0117015
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0117134, upper bound: 0.0117015
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0117134, upper bound: 0.0117016
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0117134, upper bound: 0.0117016
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0113937, upper bound: 0.0126390
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0113937, upper bound: 0.0126390
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0113937, upper bound: 0.0126390
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0113937, upper bound: 0.0126391
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0113338, upper bound: 0.0125072
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0113338, upper bound: 0.0125072
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0113338, upper bound: 0.0125072
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0113338, upper bound: 0.0125072
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0113124, upper bound: 0.0125004
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0113124, upper bound: 0.0125004
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0113124, upper bound: 0.0125004
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0113124, upper bound: 0.0125004
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0113123, upper bound: 0.0125004
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0113123, upper bound: 0.0125004
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0113123, upper bound: 0.0125004
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0113123, upper bound: 0.0125004
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0126390, upper bound: 0.0113938
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0126390, upper bound: 0.0113938
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0126390, upper bound: 0.0114116
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0126390, upper bound: 0.0114116
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113123
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113123
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113322
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113322
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113143
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113142
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113310
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113310
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113123
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113123
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113310
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113310
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0126390, upper bound: 0.0113938
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0126390, upper bound: 0.0113938
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0126390, upper bound: 0.0114116
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0126390, upper bound: 0.0114116
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113123
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113123
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113322
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125114, upper bound: 0.0113322
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113143
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113143
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113310
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125257, upper bound: 0.0113310
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113123
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113123
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113310
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0113310

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004521, 0.0009263, -0.0004521, 0.0009263, -0.0010261, 0.0010261
1: -0.0007963, 0.0027411, -0.0007963, 0.0027411, -0.0025854, 0.0025854
2: 0.0122348, 0.0175325, 0.0122348, 0.0175325, -0.0036168, 0.0036168
3: -0.0014268, 0.0025568, -0.0014268, 0.0025568, -0.0026066, 0.0026066
4: -0.0056957, -0.0020212, -0.0056957, -0.0020212, -0.0033337, 0.0033337
5: 0.0065139, 0.0104903, 0.0065139, 0.0104903, -0.0025916, 0.0025916
6: 0.0077211, 0.0104747, 0.0077211, 0.0104747, -0.0027536, 0.0027536
7: -0.0211728, -0.0125405, -0.0211728, -0.0125405, -0.0048328, 0.0048328
8: 0.9631283, 0.9878607, 0.9631283, 0.9878607, -0.0171464, 0.0171464
9: 0.0012209, 0.0084898, 0.0012209, 0.0084898, -0.0042389, 0.0042389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116466, upper bound: 0.0117010
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116570, upper bound: 0.0116678
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004521, 0.0009263, -0.0004639, 0.0009006, -0.0010032, 0.0010439
1: -0.0007963, 0.0027411, -0.0008517, 0.0027019, -0.0025690, 0.0026213
2: 0.0122348, 0.0175325, 0.0122936, 0.0176155, -0.0036868, 0.0035923
3: -0.0014268, 0.0025568, -0.0013826, 0.0026192, -0.0026697, 0.0025881
4: -0.0056957, -0.0020212, -0.0056549, -0.0019637, -0.0033267, 0.0033166
5: 0.0065139, 0.0104903, 0.0065581, 0.0105526, -0.0026557, 0.0025731
6: 0.0077211, 0.0104747, 0.0077807, 0.0104581, -0.0027369, 0.0026940
7: -0.0211728, -0.0125405, -0.0213079, -0.0126364, -0.0047927, 0.0050311
8: 0.9631283, 0.9878607, 0.9627411, 0.9875861, -0.0170316, 0.0174516
9: 0.0012209, 0.0084898, 0.0013016, 0.0086036, -0.0044065, 0.0042051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116466, upper bound: 0.0117010
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116570, upper bound: 0.0116678
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004639, 0.0009006, -0.0004521, 0.0009263, -0.0010439, 0.0010032
1: -0.0008517, 0.0027019, -0.0007963, 0.0027411, -0.0026213, 0.0025690
2: 0.0122936, 0.0176155, 0.0122348, 0.0175325, -0.0035923, 0.0036868
3: -0.0013826, 0.0026192, -0.0014268, 0.0025568, -0.0025881, 0.0026697
4: -0.0056549, -0.0019637, -0.0056957, -0.0020212, -0.0033166, 0.0033267
5: 0.0065581, 0.0105526, 0.0065139, 0.0104903, -0.0025731, 0.0026557
6: 0.0077807, 0.0104581, 0.0077211, 0.0104747, -0.0026940, 0.0027369
7: -0.0213079, -0.0126364, -0.0211728, -0.0125405, -0.0050311, 0.0047927
8: 0.9627411, 0.9875861, 0.9631283, 0.9878607, -0.0174516, 0.0170316
9: 0.0013016, 0.0086036, 0.0012209, 0.0084898, -0.0042051, 0.0044065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116466, upper bound: 0.0116860
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116570, upper bound: 0.0116570
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004639, 0.0009006, -0.0004639, 0.0009006, -0.0010123, 0.0010123
1: -0.0008517, 0.0027019, -0.0008517, 0.0027019, -0.0025547, 0.0025547
2: 0.0122936, 0.0176155, 0.0122936, 0.0176155, -0.0035814, 0.0035814
3: -0.0013826, 0.0026192, -0.0013826, 0.0026192, -0.0025874, 0.0025874
4: -0.0056549, -0.0019637, -0.0056549, -0.0019637, -0.0032782, 0.0032782
5: 0.0065581, 0.0105526, 0.0065581, 0.0105526, -0.0025731, 0.0025731
6: 0.0077807, 0.0104581, 0.0077807, 0.0104581, -0.0026773, 0.0026773
7: -0.0213079, -0.0126364, -0.0213079, -0.0126364, -0.0048610, 0.0048610
8: 0.9627411, 0.9875861, 0.9627411, 0.9875861, -0.0169690, 0.0169690
9: 0.0013016, 0.0086036, 0.0013016, 0.0086036, -0.0042473, 0.0042473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116466, upper bound: 0.0116860
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116570, upper bound: 0.0116570
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004521, 0.0009263, -0.0004370, 0.0009720, -0.0010781, 0.0010222
1: -0.0007963, 0.0027411, -0.0007255, 0.0028113, -0.0027147, 0.0026294
2: 0.0122348, 0.0175325, 0.0121298, 0.0174265, -0.0036755, 0.0038105
3: -0.0014268, 0.0025568, -0.0015058, 0.0024771, -0.0026500, 0.0027522
4: -0.0056957, -0.0020212, -0.0057686, -0.0020947, -0.0034013, 0.0034680
5: 0.0065139, 0.0104903, 0.0064351, 0.0104108, -0.0026349, 0.0027369
6: 0.0077211, 0.0104747, 0.0076148, 0.0105045, -0.0027833, 0.0028600
7: -0.0211728, -0.0125405, -0.0210001, -0.0123694, -0.0051483, 0.0048102
8: 0.9631283, 0.9878607, 0.9636232, 0.9883512, -0.0180504, 0.0174208
9: 0.0012209, 0.0084898, 0.0010767, 0.0083443, -0.0042668, 0.0045046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115797, upper bound: 0.0116308
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115917, upper bound: 0.0116022
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004521, 0.0009263, -0.0004493, 0.0009464, -0.0010585, 0.0010394
1: -0.0007963, 0.0027411, -0.0007832, 0.0027720, -0.0026960, 0.0026603
2: 0.0122348, 0.0175325, 0.0121886, 0.0175129, -0.0037296, 0.0037824
3: -0.0014268, 0.0025568, -0.0014616, 0.0025421, -0.0026943, 0.0027311
4: -0.0056957, -0.0020212, -0.0057278, -0.0020348, -0.0033914, 0.0034485
5: 0.0065139, 0.0104903, 0.0064792, 0.0104756, -0.0026791, 0.0027158
6: 0.0077211, 0.0104747, 0.0076743, 0.0104878, -0.0027667, 0.0028004
7: -0.0211728, -0.0125405, -0.0211408, -0.0124652, -0.0051025, 0.0050064
8: 0.9631283, 0.9878607, 0.9632200, 0.9880767, -0.0179192, 0.0176699
9: 0.0012209, 0.0084898, 0.0011574, 0.0084628, -0.0044129, 0.0044660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115797, upper bound: 0.0116308
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115917, upper bound: 0.0116022
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004639, 0.0009006, -0.0004370, 0.0009720, -0.0010959, 0.0009993
1: -0.0008517, 0.0027019, -0.0007255, 0.0028113, -0.0027506, 0.0026129
2: 0.0122936, 0.0176155, 0.0121298, 0.0174265, -0.0036509, 0.0038804
3: -0.0013826, 0.0026192, -0.0015058, 0.0024771, -0.0026315, 0.0028153
4: -0.0056549, -0.0019637, -0.0057686, -0.0020947, -0.0033842, 0.0034610
5: 0.0065581, 0.0105526, 0.0064351, 0.0104108, -0.0026164, 0.0028010
6: 0.0077807, 0.0104581, 0.0076148, 0.0105045, -0.0027238, 0.0028433
7: -0.0213079, -0.0126364, -0.0210001, -0.0123694, -0.0053466, 0.0047701
8: 0.9627411, 0.9875861, 0.9636232, 0.9883512, -0.0183556, 0.0173060
9: 0.0013016, 0.0086036, 0.0010767, 0.0083443, -0.0042331, 0.0046722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115797, upper bound: 0.0116121
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115917, upper bound: 0.0116010
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004639, 0.0009006, -0.0004493, 0.0009464, -0.0010662, 0.0010082
1: -0.0008517, 0.0027019, -0.0007832, 0.0027720, -0.0026843, 0.0026009
2: 0.0122936, 0.0176155, 0.0121886, 0.0175129, -0.0036466, 0.0037756
3: -0.0013826, 0.0026192, -0.0014616, 0.0025421, -0.0026345, 0.0027334
4: -0.0056549, -0.0019637, -0.0057278, -0.0020348, -0.0033481, 0.0034129
5: 0.0065581, 0.0105526, 0.0064792, 0.0104756, -0.0026198, 0.0027188
6: 0.0077807, 0.0104581, 0.0076743, 0.0104878, -0.0027071, 0.0027837
7: -0.0213079, -0.0126364, -0.0211408, -0.0124652, -0.0051773, 0.0048367
8: 0.9627411, 0.9875861, 0.9632200, 0.9880767, -0.0178754, 0.0172791
9: 0.0013016, 0.0086036, 0.0011574, 0.0084628, -0.0042690, 0.0045137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115797, upper bound: 0.0116121
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115917, upper bound: 0.0116010
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004370, 0.0009720, -0.0004521, 0.0009263, -0.0010222, 0.0010781
1: -0.0007255, 0.0028113, -0.0007963, 0.0027411, -0.0026294, 0.0027147
2: 0.0121298, 0.0174265, 0.0122348, 0.0175325, -0.0038105, 0.0036755
3: -0.0015058, 0.0024771, -0.0014268, 0.0025568, -0.0027522, 0.0026500
4: -0.0057686, -0.0020947, -0.0056957, -0.0020212, -0.0034680, 0.0034013
5: 0.0064351, 0.0104108, 0.0065139, 0.0104903, -0.0027369, 0.0026349
6: 0.0076148, 0.0105045, 0.0077211, 0.0104747, -0.0028600, 0.0027833
7: -0.0210001, -0.0123694, -0.0211728, -0.0125405, -0.0048102, 0.0051483
8: 0.9636232, 0.9883512, 0.9631283, 0.9878607, -0.0174208, 0.0180504
9: 0.0010767, 0.0083443, 0.0012209, 0.0084898, -0.0045046, 0.0042668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112693, upper bound: 0.0114289
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112581, upper bound: 0.0112581
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004370, 0.0009720, -0.0004370, 0.0009720, -0.0010597, 0.0010597
1: -0.0007255, 0.0028113, -0.0007255, 0.0028113, -0.0026637, 0.0026637
2: 0.0121298, 0.0174265, 0.0121298, 0.0174265, -0.0037174, 0.0037174
3: -0.0015058, 0.0024771, -0.0015058, 0.0024771, -0.0026754, 0.0026754
4: -0.0057686, -0.0020947, -0.0057686, -0.0020947, -0.0034619, 0.0034619
5: 0.0064351, 0.0104108, 0.0064351, 0.0104108, -0.0026592, 0.0026592
6: 0.0076148, 0.0105045, 0.0076148, 0.0105045, -0.0028897, 0.0028897
7: -0.0210001, -0.0123694, -0.0210001, -0.0123694, -0.0048797, 0.0048797
8: 0.9636232, 0.9883512, 0.9636232, 0.9883512, -0.0176303, 0.0176303
9: 0.0010767, 0.0083443, 0.0010767, 0.0083443, -0.0042976, 0.0042976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112693, upper bound: 0.0114289
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112581, upper bound: 0.0112581
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004493, 0.0009464, -0.0004521, 0.0009263, -0.0010394, 0.0010585
1: -0.0007832, 0.0027720, -0.0007963, 0.0027411, -0.0026603, 0.0026960
2: 0.0121886, 0.0175129, 0.0122348, 0.0175325, -0.0037824, 0.0037296
3: -0.0014616, 0.0025421, -0.0014268, 0.0025568, -0.0027311, 0.0026943
4: -0.0057278, -0.0020348, -0.0056957, -0.0020212, -0.0034485, 0.0033914
5: 0.0064792, 0.0104756, 0.0065139, 0.0104903, -0.0027158, 0.0026791
6: 0.0076743, 0.0104878, 0.0077211, 0.0104747, -0.0028004, 0.0027667
7: -0.0211408, -0.0124652, -0.0211728, -0.0125405, -0.0050064, 0.0051025
8: 0.9632200, 0.9880767, 0.9631283, 0.9878607, -0.0176699, 0.0179192
9: 0.0011574, 0.0084628, 0.0012209, 0.0084898, -0.0044660, 0.0044129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116008, upper bound: 0.0116058
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116022, upper bound: 0.0115806
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004493, 0.0009464, -0.0004370, 0.0009720, -0.0010770, 0.0010398
1: -0.0007832, 0.0027720, -0.0007255, 0.0028113, -0.0026974, 0.0026472
2: 0.0121886, 0.0175129, 0.0121298, 0.0174265, -0.0036927, 0.0037824
3: -0.0014616, 0.0025421, -0.0015058, 0.0024771, -0.0026568, 0.0027335
4: -0.0057278, -0.0020348, -0.0057686, -0.0020947, -0.0034448, 0.0034510
5: 0.0064792, 0.0104756, 0.0064351, 0.0104108, -0.0026406, 0.0027180
6: 0.0076743, 0.0104878, 0.0076148, 0.0105045, -0.0028302, 0.0028731
7: -0.0211408, -0.0124652, -0.0210001, -0.0123694, -0.0050733, 0.0048395
8: 0.9632200, 0.9880767, 0.9636232, 0.9883512, -0.0179218, 0.0175150
9: 0.0011574, 0.0084628, 0.0010767, 0.0083443, -0.0042637, 0.0044664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116008, upper bound: 0.0116058
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116022, upper bound: 0.0115806
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004370, 0.0009720, -0.0004639, 0.0009006, -0.0009993, 0.0010959
1: -0.0007255, 0.0028113, -0.0008517, 0.0027019, -0.0026129, 0.0027506
2: 0.0121298, 0.0174265, 0.0122936, 0.0176155, -0.0038804, 0.0036509
3: -0.0015058, 0.0024771, -0.0013826, 0.0026192, -0.0028153, 0.0026315
4: -0.0057686, -0.0020947, -0.0056549, -0.0019637, -0.0034610, 0.0033842
5: 0.0064351, 0.0104108, 0.0065581, 0.0105526, -0.0028010, 0.0026164
6: 0.0076148, 0.0105045, 0.0077807, 0.0104581, -0.0028433, 0.0027238
7: -0.0210001, -0.0123694, -0.0213079, -0.0126364, -0.0047701, 0.0053466
8: 0.9636232, 0.9883512, 0.9627411, 0.9875861, -0.0173060, 0.0183556
9: 0.0010767, 0.0083443, 0.0013016, 0.0086036, -0.0046722, 0.0042331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109072, upper bound: 0.0111352
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108064, upper bound: 0.0108669
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004370, 0.0009720, -0.0004493, 0.0009464, -0.0010398, 0.0010770
1: -0.0007255, 0.0028113, -0.0007832, 0.0027720, -0.0026472, 0.0026974
2: 0.0121298, 0.0174265, 0.0121886, 0.0175129, -0.0037824, 0.0036927
3: -0.0015058, 0.0024771, -0.0014616, 0.0025421, -0.0027335, 0.0026568
4: -0.0057686, -0.0020947, -0.0057278, -0.0020348, -0.0034510, 0.0034448
5: 0.0064351, 0.0104108, 0.0064792, 0.0104756, -0.0027180, 0.0026406
6: 0.0076148, 0.0105045, 0.0076743, 0.0104878, -0.0028731, 0.0028302
7: -0.0210001, -0.0123694, -0.0211408, -0.0124652, -0.0048395, 0.0050733
8: 0.9636232, 0.9883512, 0.9632200, 0.9880767, -0.0175150, 0.0179218
9: 0.0010767, 0.0083443, 0.0011574, 0.0084628, -0.0044664, 0.0042637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109072, upper bound: 0.0111352
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108064, upper bound: 0.0108669
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004493, 0.0009464, -0.0004639, 0.0009006, -0.0010082, 0.0010662
1: -0.0007832, 0.0027720, -0.0008517, 0.0027019, -0.0026009, 0.0026843
2: 0.0121886, 0.0175129, 0.0122936, 0.0176155, -0.0037756, 0.0036466
3: -0.0014616, 0.0025421, -0.0013826, 0.0026192, -0.0027334, 0.0026345
4: -0.0057278, -0.0020348, -0.0056549, -0.0019637, -0.0034129, 0.0033481
5: 0.0064792, 0.0104756, 0.0065581, 0.0105526, -0.0027188, 0.0026198
6: 0.0076743, 0.0104878, 0.0077807, 0.0104581, -0.0027837, 0.0027071
7: -0.0211408, -0.0124652, -0.0213079, -0.0126364, -0.0048367, 0.0051773
8: 0.9632200, 0.9880767, 0.9627411, 0.9875861, -0.0172791, 0.0178754
9: 0.0011574, 0.0084628, 0.0013016, 0.0086036, -0.0045137, 0.0042690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115987, upper bound: 0.0116058
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115998, upper bound: 0.0115806
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004493, 0.0009464, -0.0004493, 0.0009464, -0.0010472, 0.0010472
1: -0.0007832, 0.0027720, -0.0007832, 0.0027720, -0.0026325, 0.0026325
2: 0.0121886, 0.0175129, 0.0121886, 0.0175129, -0.0036824, 0.0036824
3: -0.0014616, 0.0025421, -0.0014616, 0.0025421, -0.0026549, 0.0026549
4: -0.0057278, -0.0020348, -0.0057278, -0.0020348, -0.0034049, 0.0034049
5: 0.0064792, 0.0104756, 0.0064792, 0.0104756, -0.0026393, 0.0026393
6: 0.0076743, 0.0104878, 0.0076743, 0.0104878, -0.0028135, 0.0028135
7: -0.0211408, -0.0124652, -0.0211408, -0.0124652, -0.0049073, 0.0049073
8: 0.9632200, 0.9880767, 0.9632200, 0.9880767, -0.0174561, 0.0174561
9: 0.0011574, 0.0084628, 0.0011574, 0.0084628, -0.0043045, 0.0043045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115987, upper bound: 0.0116058
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115998, upper bound: 0.0115806
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004521, 0.0009263, -0.0004879, 0.0012858, -0.0014025, 0.0010822
1: -0.0007963, 0.0027411, -0.0009639, 0.0032921, -0.0032336, 0.0028838
2: 0.0122348, 0.0175325, 0.0114097, 0.0177836, -0.0040282, 0.0045876
3: -0.0014268, 0.0025568, -0.0020473, 0.0027456, -0.0029126, 0.0033365
4: -0.0056957, -0.0020212, -0.0062681, -0.0018471, -0.0037545, 0.0040070
5: 0.0065139, 0.0104903, 0.0058946, 0.0106788, -0.0028975, 0.0033202
6: 0.0077211, 0.0104747, 0.0068855, 0.0107084, -0.0029873, 0.0035892
7: -0.0211728, -0.0125405, -0.0215819, -0.0111960, -0.0064145, 0.0055132
8: 0.9631283, 0.9878607, 0.9619563, 0.9917130, -0.0216783, 0.0190992
9: 0.0012209, 0.0084898, 0.0000887, 0.0088343, -0.0048272, 0.0055708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112631, upper bound: 0.0125512
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112738, upper bound: 0.0125316
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004521, 0.0009263, -0.0005002, 0.0012606, -0.0013808, 0.0010961
1: -0.0007963, 0.0027411, -0.0010216, 0.0032535, -0.0032026, 0.0029158
2: 0.0122348, 0.0175325, 0.0114675, 0.0178700, -0.0041040, 0.0045411
3: -0.0014268, 0.0025568, -0.0020039, 0.0028106, -0.0029753, 0.0033016
4: -0.0056957, -0.0020212, -0.0062280, -0.0017872, -0.0037538, 0.0039748
5: 0.0065139, 0.0104903, 0.0059380, 0.0107436, -0.0029599, 0.0032854
6: 0.0077211, 0.0104747, 0.0069440, 0.0106921, -0.0029709, 0.0035307
7: -0.0211728, -0.0125405, -0.0217226, -0.0112902, -0.0063388, 0.0056646
8: 0.9631283, 0.9878607, 0.9615530, 0.9914432, -0.0214614, 0.0194199
9: 0.0012209, 0.0084898, 0.0001680, 0.0089527, -0.0049588, 0.0055070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112631, upper bound: 0.0125512
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112738, upper bound: 0.0125316
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004639, 0.0009006, -0.0004879, 0.0012858, -0.0014204, 0.0010594
1: -0.0008517, 0.0027019, -0.0009639, 0.0032921, -0.0032695, 0.0028674
2: 0.0122936, 0.0176155, 0.0114097, 0.0177836, -0.0040036, 0.0046575
3: -0.0013826, 0.0026192, -0.0020473, 0.0027456, -0.0028941, 0.0033996
4: -0.0056549, -0.0019637, -0.0062681, -0.0018471, -0.0037375, 0.0040000
5: 0.0065581, 0.0105526, 0.0058946, 0.0106788, -0.0028791, 0.0033843
6: 0.0077807, 0.0104581, 0.0068855, 0.0107084, -0.0029277, 0.0035725
7: -0.0213079, -0.0126364, -0.0215819, -0.0111960, -0.0066128, 0.0054732
8: 0.9627411, 0.9875861, 0.9619563, 0.9917130, -0.0219834, 0.0189844
9: 0.0013016, 0.0086036, 0.0000887, 0.0088343, -0.0047934, 0.0057384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112616, upper bound: 0.0125321
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112721, upper bound: 0.0125114
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004639, 0.0009006, -0.0005002, 0.0012606, -0.0013903, 0.0010678
1: -0.0008517, 0.0027019, -0.0010216, 0.0032535, -0.0032018, 0.0028617
2: 0.0122936, 0.0176155, 0.0114675, 0.0178700, -0.0040030, 0.0045506
3: -0.0013826, 0.0026192, -0.0020039, 0.0028106, -0.0029039, 0.0033162
4: -0.0056549, -0.0019637, -0.0062280, -0.0017872, -0.0037164, 0.0039505
5: 0.0065581, 0.0105526, 0.0059380, 0.0107436, -0.0028893, 0.0033006
6: 0.0077807, 0.0104581, 0.0069440, 0.0106921, -0.0029114, 0.0035140
7: -0.0213079, -0.0126364, -0.0217226, -0.0112902, -0.0064402, 0.0055374
8: 0.9627411, 0.9875861, 0.9615530, 0.9914432, -0.0214937, 0.0189728
9: 0.0013016, 0.0086036, 0.0001680, 0.0089527, -0.0048427, 0.0055771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112616, upper bound: 0.0125321
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112721, upper bound: 0.0125114
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004521, 0.0009263, -0.0004742, 0.0013369, -0.0014587, 0.0010740
1: -0.0007963, 0.0027411, -0.0008998, 0.0033706, -0.0033465, 0.0029042
2: 0.0122348, 0.0175325, 0.0112922, 0.0176875, -0.0040485, 0.0047566
3: -0.0014268, 0.0025568, -0.0021356, 0.0026734, -0.0029173, 0.0034636
4: -0.0056957, -0.0020212, -0.0063495, -0.0019137, -0.0037820, 0.0041242
5: 0.0065139, 0.0104903, 0.0058064, 0.0106067, -0.0029003, 0.0034470
6: 0.0077211, 0.0104747, 0.0067666, 0.0107417, -0.0030206, 0.0037081
7: -0.0211728, -0.0125405, -0.0214253, -0.0110046, -0.0066898, 0.0054254
8: 0.9631283, 0.9878607, 0.9624048, 0.9922614, -0.0224671, 0.0192038
9: 0.0012209, 0.0084898, -0.0000725, 0.0087024, -0.0047687, 0.0058026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111991, upper bound: 0.0124227
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112066, upper bound: 0.0123993
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004521, 0.0009263, -0.0004871, 0.0013133, -0.0014381, 0.0010872
1: -0.0007963, 0.0027411, -0.0009600, 0.0033343, -0.0033115, 0.0029325
2: 0.0122348, 0.0175325, 0.0113465, 0.0177776, -0.0041043, 0.0047043
3: -0.0014268, 0.0025568, -0.0020948, 0.0027411, -0.0029643, 0.0034243
4: -0.0056957, -0.0020212, -0.0063119, -0.0018512, -0.0038051, 0.0040879
5: 0.0065139, 0.0104903, 0.0058471, 0.0106743, -0.0029483, 0.0034078
6: 0.0077211, 0.0104747, 0.0068215, 0.0107263, -0.0030052, 0.0036532
7: -0.0211728, -0.0125405, -0.0215721, -0.0110930, -0.0066047, 0.0055794
8: 0.9631283, 0.9878607, 0.9619842, 0.9920081, -0.0222231, 0.0194467
9: 0.0012209, 0.0084898, 0.0000019, 0.0088260, -0.0049057, 0.0057309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111991, upper bound: 0.0124227
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112066, upper bound: 0.0123993
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004639, 0.0009006, -0.0004742, 0.0013369, -0.0014766, 0.0010511
1: -0.0008517, 0.0027019, -0.0008998, 0.0033706, -0.0033823, 0.0028878
2: 0.0122936, 0.0176155, 0.0112922, 0.0176875, -0.0040239, 0.0048265
3: -0.0013826, 0.0026192, -0.0021356, 0.0026734, -0.0028988, 0.0035267
4: -0.0056549, -0.0019637, -0.0063495, -0.0019137, -0.0037412, 0.0041172
5: 0.0065581, 0.0105526, 0.0058064, 0.0106067, -0.0028819, 0.0035111
6: 0.0077807, 0.0104581, 0.0067666, 0.0107417, -0.0029610, 0.0036915
7: -0.0213079, -0.0126364, -0.0214253, -0.0110046, -0.0068881, 0.0053853
8: 0.9627411, 0.9875861, 0.9624048, 0.9922614, -0.0227723, 0.0190891
9: 0.0013016, 0.0086036, -0.0000725, 0.0087024, -0.0047350, 0.0059702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111983, upper bound: 0.0124003
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112051, upper bound: 0.0123868
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004639, 0.0009006, -0.0004871, 0.0013133, -0.0014477, 0.0010594
1: -0.0008517, 0.0027019, -0.0009600, 0.0033343, -0.0033150, 0.0028864
2: 0.0122936, 0.0176155, 0.0113465, 0.0177776, -0.0040314, 0.0047201
3: -0.0013826, 0.0026192, -0.0020948, 0.0027411, -0.0029105, 0.0034436
4: -0.0056549, -0.0019637, -0.0063119, -0.0018512, -0.0037701, 0.0040680
5: 0.0065581, 0.0105526, 0.0058471, 0.0106743, -0.0028940, 0.0034278
6: 0.0077807, 0.0104581, 0.0068215, 0.0107263, -0.0029456, 0.0036366
7: -0.0213079, -0.0126364, -0.0215721, -0.0110930, -0.0067163, 0.0054507
8: 0.9627411, 0.9875861, 0.9619842, 0.9920081, -0.0222847, 0.0191100
9: 0.0013016, 0.0086036, 0.0000019, 0.0088260, -0.0047849, 0.0058096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111983, upper bound: 0.0124003
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112051, upper bound: 0.0123868
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004370, 0.0009720, -0.0004879, 0.0012858, -0.0013986, 0.0011343
1: -0.0007255, 0.0028113, -0.0009639, 0.0032921, -0.0032775, 0.0030131
2: 0.0121298, 0.0174265, 0.0114097, 0.0177836, -0.0042218, 0.0046463
3: -0.0015058, 0.0024771, -0.0020473, 0.0027456, -0.0030582, 0.0033799
4: -0.0057686, -0.0020947, -0.0062681, -0.0018471, -0.0038888, 0.0040746
5: 0.0064351, 0.0104108, 0.0058946, 0.0106788, -0.0030429, 0.0033635
6: 0.0076148, 0.0105045, 0.0068855, 0.0107084, -0.0030937, 0.0036190
7: -0.0210001, -0.0123694, -0.0215819, -0.0111960, -0.0063919, 0.0058288
8: 0.9636232, 0.9883512, 0.9619563, 0.9917130, -0.0219526, 0.0200032
9: 0.0010767, 0.0083443, 0.0000887, 0.0088343, -0.0050929, 0.0055988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109319, upper bound: 0.0119012
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108288, upper bound: 0.0116354
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004370, 0.0009720, -0.0004742, 0.0013369, -0.0014403, 0.0011150
1: -0.0007255, 0.0028113, -0.0008998, 0.0033706, -0.0033062, 0.0029491
2: 0.0121298, 0.0174265, 0.0112922, 0.0176875, -0.0041109, 0.0046796
3: -0.0015058, 0.0024771, -0.0021356, 0.0026734, -0.0029626, 0.0033990
4: -0.0057686, -0.0020947, -0.0063495, -0.0019137, -0.0038549, 0.0041293
5: 0.0064351, 0.0104108, 0.0058064, 0.0106067, -0.0029458, 0.0033815
6: 0.0076148, 0.0105045, 0.0067666, 0.0107417, -0.0031269, 0.0037379
7: -0.0210001, -0.0123694, -0.0214253, -0.0110046, -0.0064477, 0.0055504
8: 0.9636232, 0.9883512, 0.9624048, 0.9922614, -0.0221226, 0.0195003
9: 0.0010767, 0.0083443, -0.0000725, 0.0087024, -0.0048662, 0.0056179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109319, upper bound: 0.0119012
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108288, upper bound: 0.0116354
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004493, 0.0009464, -0.0004879, 0.0012858, -0.0014158, 0.0011147
1: -0.0007832, 0.0027720, -0.0009639, 0.0032921, -0.0033085, 0.0029944
2: 0.0121886, 0.0175129, 0.0114097, 0.0177836, -0.0041937, 0.0047003
3: -0.0014616, 0.0025421, -0.0020473, 0.0027456, -0.0030371, 0.0034242
4: -0.0057278, -0.0020348, -0.0062681, -0.0018471, -0.0038693, 0.0040647
5: 0.0064792, 0.0104756, 0.0058946, 0.0106788, -0.0030218, 0.0034077
6: 0.0076743, 0.0104878, 0.0068855, 0.0107084, -0.0030341, 0.0036023
7: -0.0211408, -0.0124652, -0.0215819, -0.0111960, -0.0065881, 0.0057830
8: 0.9632200, 0.9880767, 0.9619563, 0.9917130, -0.0222017, 0.0198720
9: 0.0011574, 0.0084628, 0.0000887, 0.0088343, -0.0050543, 0.0057448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111887, upper bound: 0.0123973
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111889, upper bound: 0.0123751
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004493, 0.0009464, -0.0004742, 0.0013369, -0.0014576, 0.0010951
1: -0.0007832, 0.0027720, -0.0008998, 0.0033706, -0.0033399, 0.0029326
2: 0.0121886, 0.0175129, 0.0112922, 0.0176875, -0.0040862, 0.0047447
3: -0.0014616, 0.0025421, -0.0021356, 0.0026734, -0.0029440, 0.0034571
4: -0.0057278, -0.0020348, -0.0063495, -0.0019137, -0.0038141, 0.0041185
5: 0.0064792, 0.0104756, 0.0058064, 0.0106067, -0.0029273, 0.0034403
6: 0.0076743, 0.0104878, 0.0067666, 0.0107417, -0.0030674, 0.0037212
7: -0.0211408, -0.0124652, -0.0214253, -0.0110046, -0.0066412, 0.0055102
8: 0.9632200, 0.9880767, 0.9624048, 0.9922614, -0.0224141, 0.0193850
9: 0.0011574, 0.0084628, -0.0000725, 0.0087024, -0.0048323, 0.0057867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111887, upper bound: 0.0123973
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111889, upper bound: 0.0123751
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004370, 0.0009720, -0.0005002, 0.0012606, -0.0013770, 0.0011482
1: -0.0007255, 0.0028113, -0.0010216, 0.0032535, -0.0032465, 0.0030451
2: 0.0121298, 0.0174265, 0.0114675, 0.0178700, -0.0042976, 0.0045998
3: -0.0015058, 0.0024771, -0.0020039, 0.0028106, -0.0031209, 0.0033450
4: -0.0057686, -0.0020947, -0.0062280, -0.0017872, -0.0038882, 0.0040424
5: 0.0064351, 0.0104108, 0.0059380, 0.0107436, -0.0031053, 0.0033286
6: 0.0076148, 0.0105045, 0.0069440, 0.0106921, -0.0030773, 0.0035604
7: -0.0210001, -0.0123694, -0.0217226, -0.0112902, -0.0063162, 0.0059801
8: 0.9636232, 0.9883512, 0.9615530, 0.9914432, -0.0217358, 0.0203239
9: 0.0010767, 0.0083443, 0.0001680, 0.0089527, -0.0052245, 0.0055350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105785, upper bound: 0.0114709
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103882, upper bound: 0.0111206
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004370, 0.0009720, -0.0004871, 0.0013133, -0.0014195, 0.0011284
1: -0.0007255, 0.0028113, -0.0009600, 0.0033343, -0.0032744, 0.0029797
2: 0.0121298, 0.0174265, 0.0113465, 0.0177776, -0.0041826, 0.0046320
3: -0.0015058, 0.0024771, -0.0020948, 0.0027411, -0.0030271, 0.0033631
4: -0.0057686, -0.0020947, -0.0063119, -0.0018512, -0.0038726, 0.0040963
5: 0.0064351, 0.0104108, 0.0058471, 0.0106743, -0.0030106, 0.0033457
6: 0.0076148, 0.0105045, 0.0068215, 0.0107263, -0.0031116, 0.0036830
7: -0.0210001, -0.0123694, -0.0215721, -0.0110930, -0.0063700, 0.0057009
8: 0.9636232, 0.9883512, 0.9619842, 0.9920081, -0.0219001, 0.0198101
9: 0.0010767, 0.0083443, 0.0000019, 0.0088260, -0.0049992, 0.0055525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105785, upper bound: 0.0114709
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103882, upper bound: 0.0111206
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004493, 0.0009464, -0.0005002, 0.0012606, -0.0013861, 0.0011217
1: -0.0007832, 0.0027720, -0.0010216, 0.0032535, -0.0032481, 0.0029913
2: 0.0121886, 0.0175129, 0.0114675, 0.0178700, -0.0041971, 0.0046158
3: -0.0014616, 0.0025421, -0.0020039, 0.0028106, -0.0030499, 0.0033633
4: -0.0057278, -0.0020348, -0.0062280, -0.0017872, -0.0038511, 0.0040204
5: 0.0064792, 0.0104756, 0.0059380, 0.0107436, -0.0030350, 0.0033473
6: 0.0076743, 0.0104878, 0.0069440, 0.0106921, -0.0030178, 0.0035438
7: -0.0211408, -0.0124652, -0.0217226, -0.0112902, -0.0064159, 0.0058538
8: 0.9632200, 0.9880767, 0.9615530, 0.9914432, -0.0218039, 0.0198792
9: 0.0011574, 0.0084628, 0.0001680, 0.0089527, -0.0051091, 0.0055988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111885, upper bound: 0.0123973
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111886, upper bound: 0.0123751
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004493, 0.0009464, -0.0004871, 0.0013133, -0.0014289, 0.0011017
1: -0.0007832, 0.0027720, -0.0009600, 0.0033343, -0.0032741, 0.0029301
2: 0.0121886, 0.0175129, 0.0113465, 0.0177776, -0.0040870, 0.0046432
3: -0.0014616, 0.0025421, -0.0020948, 0.0027411, -0.0029502, 0.0033774
4: -0.0057278, -0.0020348, -0.0063119, -0.0018512, -0.0038337, 0.0040714
5: 0.0064792, 0.0104756, 0.0058471, 0.0106743, -0.0029348, 0.0033605
6: 0.0076743, 0.0104878, 0.0068215, 0.0107263, -0.0030520, 0.0036663
7: -0.0211408, -0.0124652, -0.0215721, -0.0110930, -0.0064729, 0.0055748
8: 0.9632200, 0.9880767, 0.9619842, 0.9920081, -0.0219418, 0.0193814
9: 0.0011574, 0.0084628, 0.0000019, 0.0088260, -0.0048816, 0.0056229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111885, upper bound: 0.0123973
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111886, upper bound: 0.0123751
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004879, 0.0012858, -0.0004521, 0.0009263, -0.0010822, 0.0014025
1: -0.0009639, 0.0032921, -0.0007963, 0.0027411, -0.0028838, 0.0032336
2: 0.0114097, 0.0177836, 0.0122348, 0.0175325, -0.0045876, 0.0040282
3: -0.0020473, 0.0027456, -0.0014268, 0.0025568, -0.0033365, 0.0029126
4: -0.0062681, -0.0018471, -0.0056957, -0.0020212, -0.0040070, 0.0037545
5: 0.0058946, 0.0106788, 0.0065139, 0.0104903, -0.0033202, 0.0028975
6: 0.0068855, 0.0107084, 0.0077211, 0.0104747, -0.0035892, 0.0029873
7: -0.0215819, -0.0111960, -0.0211728, -0.0125405, -0.0055132, 0.0064145
8: 0.9619563, 0.9917130, 0.9631283, 0.9878607, -0.0190992, 0.0216783
9: 0.0000887, 0.0088343, 0.0012209, 0.0084898, -0.0055708, 0.0048272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119233, upper bound: 0.0110759
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119101, upper bound: 0.0110625
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004879, 0.0012858, -0.0004639, 0.0009006, -0.0010594, 0.0014204
1: -0.0009639, 0.0032921, -0.0008517, 0.0027019, -0.0028674, 0.0032695
2: 0.0114097, 0.0177836, 0.0122936, 0.0176155, -0.0046575, 0.0040036
3: -0.0020473, 0.0027456, -0.0013826, 0.0026192, -0.0033996, 0.0028941
4: -0.0062681, -0.0018471, -0.0056549, -0.0019637, -0.0040000, 0.0037375
5: 0.0058946, 0.0106788, 0.0065581, 0.0105526, -0.0033843, 0.0028791
6: 0.0068855, 0.0107084, 0.0077807, 0.0104581, -0.0035725, 0.0029277
7: -0.0215819, -0.0111960, -0.0213079, -0.0126364, -0.0054732, 0.0066128
8: 0.9619563, 0.9917130, 0.9627411, 0.9875861, -0.0189844, 0.0219834
9: 0.0000887, 0.0088343, 0.0013016, 0.0086036, -0.0057384, 0.0047934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119233, upper bound: 0.0110759
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119101, upper bound: 0.0110625
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0005002, 0.0012606, -0.0004521, 0.0009263, -0.0010961, 0.0013808
1: -0.0010216, 0.0032535, -0.0007963, 0.0027411, -0.0029158, 0.0032026
2: 0.0114675, 0.0178700, 0.0122348, 0.0175325, -0.0045411, 0.0041040
3: -0.0020039, 0.0028106, -0.0014268, 0.0025568, -0.0033016, 0.0029753
4: -0.0062280, -0.0017872, -0.0056957, -0.0020212, -0.0039748, 0.0037538
5: 0.0059380, 0.0107436, 0.0065139, 0.0104903, -0.0032854, 0.0029599
6: 0.0069440, 0.0106921, 0.0077211, 0.0104747, -0.0035307, 0.0029709
7: -0.0217226, -0.0112902, -0.0211728, -0.0125405, -0.0056646, 0.0063388
8: 0.9615530, 0.9914432, 0.9631283, 0.9878607, -0.0194199, 0.0214614
9: 0.0001680, 0.0089527, 0.0012209, 0.0084898, -0.0055071, 0.0049588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119125, upper bound: 0.0110713
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118938, upper bound: 0.0110449
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0005002, 0.0012606, -0.0004639, 0.0009006, -0.0010678, 0.0013903
1: -0.0010216, 0.0032535, -0.0008517, 0.0027019, -0.0028617, 0.0032018
2: 0.0114675, 0.0178700, 0.0122936, 0.0176155, -0.0045506, 0.0040030
3: -0.0020039, 0.0028106, -0.0013826, 0.0026192, -0.0033162, 0.0029039
4: -0.0062280, -0.0017872, -0.0056549, -0.0019637, -0.0039505, 0.0037164
5: 0.0059380, 0.0107436, 0.0065581, 0.0105526, -0.0033006, 0.0028893
6: 0.0069440, 0.0106921, 0.0077807, 0.0104581, -0.0035140, 0.0029114
7: -0.0217226, -0.0112902, -0.0213079, -0.0126364, -0.0055374, 0.0064402
8: 0.9615530, 0.9914432, 0.9627411, 0.9875861, -0.0189729, 0.0214937
9: 0.0001680, 0.0089527, 0.0013016, 0.0086036, -0.0055771, 0.0048427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119125, upper bound: 0.0110713
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118938, upper bound: 0.0110449
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004879, 0.0012858, -0.0004370, 0.0009720, -0.0011343, 0.0013986
1: -0.0009639, 0.0032921, -0.0007255, 0.0028113, -0.0030131, 0.0032775
2: 0.0114097, 0.0177836, 0.0121298, 0.0174265, -0.0046463, 0.0042218
3: -0.0020473, 0.0027456, -0.0015058, 0.0024771, -0.0033799, 0.0030582
4: -0.0062681, -0.0018471, -0.0057686, -0.0020947, -0.0040746, 0.0038888
5: 0.0058946, 0.0106788, 0.0064351, 0.0104108, -0.0033635, 0.0030429
6: 0.0068855, 0.0107084, 0.0076148, 0.0105045, -0.0036190, 0.0030937
7: -0.0215819, -0.0111960, -0.0210001, -0.0123694, -0.0058288, 0.0063919
8: 0.9619563, 0.9917130, 0.9636232, 0.9883512, -0.0200032, 0.0219526
9: 0.0000887, 0.0088343, 0.0010767, 0.0083443, -0.0055988, 0.0050929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112676, upper bound: 0.0106309
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112022, upper bound: 0.0105027
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004879, 0.0012858, -0.0004493, 0.0009464, -0.0011147, 0.0014158
1: -0.0009639, 0.0032921, -0.0007832, 0.0027720, -0.0029944, 0.0033085
2: 0.0114097, 0.0177836, 0.0121886, 0.0175129, -0.0047003, 0.0041937
3: -0.0020473, 0.0027456, -0.0014616, 0.0025421, -0.0034242, 0.0030371
4: -0.0062681, -0.0018471, -0.0057278, -0.0020348, -0.0040647, 0.0038693
5: 0.0058946, 0.0106788, 0.0064792, 0.0104756, -0.0034077, 0.0030218
6: 0.0068855, 0.0107084, 0.0076743, 0.0104878, -0.0036023, 0.0030341
7: -0.0215819, -0.0111960, -0.0211408, -0.0124652, -0.0057830, 0.0065881
8: 0.9619563, 0.9917130, 0.9632200, 0.9880767, -0.0198720, 0.0222017
9: 0.0000887, 0.0088343, 0.0011574, 0.0084628, -0.0057448, 0.0050543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112676, upper bound: 0.0106309
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112022, upper bound: 0.0105027
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0005002, 0.0012606, -0.0004370, 0.0009720, -0.0011482, 0.0013770
1: -0.0010216, 0.0032535, -0.0007255, 0.0028113, -0.0030451, 0.0032465
2: 0.0114675, 0.0178700, 0.0121298, 0.0174265, -0.0045998, 0.0042976
3: -0.0020039, 0.0028106, -0.0015058, 0.0024771, -0.0033450, 0.0031209
4: -0.0062280, -0.0017872, -0.0057686, -0.0020947, -0.0040424, 0.0038881
5: 0.0059380, 0.0107436, 0.0064351, 0.0104108, -0.0033286, 0.0031053
6: 0.0069440, 0.0106921, 0.0076148, 0.0105045, -0.0035604, 0.0030773
7: -0.0217226, -0.0112902, -0.0210001, -0.0123694, -0.0059801, 0.0063162
8: 0.9615530, 0.9914432, 0.9636232, 0.9883512, -0.0203239, 0.0217358
9: 0.0001680, 0.0089527, 0.0010767, 0.0083443, -0.0055350, 0.0052245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112610, upper bound: 0.0106186
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111652, upper bound: 0.0104464
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0005002, 0.0012606, -0.0004493, 0.0009464, -0.0011217, 0.0013861
1: -0.0010216, 0.0032535, -0.0007832, 0.0027720, -0.0029913, 0.0032481
2: 0.0114675, 0.0178700, 0.0121886, 0.0175129, -0.0046158, 0.0041971
3: -0.0020039, 0.0028106, -0.0014616, 0.0025421, -0.0033633, 0.0030499
4: -0.0062280, -0.0017872, -0.0057278, -0.0020348, -0.0040204, 0.0038511
5: 0.0059380, 0.0107436, 0.0064792, 0.0104756, -0.0033473, 0.0030350
6: 0.0069440, 0.0106921, 0.0076743, 0.0104878, -0.0035438, 0.0030178
7: -0.0217226, -0.0112902, -0.0211408, -0.0124652, -0.0058538, 0.0064159
8: 0.9615530, 0.9914432, 0.9632200, 0.9880767, -0.0198792, 0.0218039
9: 0.0001680, 0.0089527, 0.0011574, 0.0084628, -0.0055988, 0.0051091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112610, upper bound: 0.0106186
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111652, upper bound: 0.0104464
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004742, 0.0013369, -0.0004521, 0.0009263, -0.0010740, 0.0014587
1: -0.0008998, 0.0033706, -0.0007963, 0.0027411, -0.0029042, 0.0033465
2: 0.0112922, 0.0176875, 0.0122348, 0.0175325, -0.0047566, 0.0040485
3: -0.0021356, 0.0026734, -0.0014268, 0.0025568, -0.0034636, 0.0029173
4: -0.0063495, -0.0019137, -0.0056957, -0.0020212, -0.0041242, 0.0037820
5: 0.0058064, 0.0106067, 0.0065139, 0.0104903, -0.0034470, 0.0029003
6: 0.0067666, 0.0107417, 0.0077211, 0.0104747, -0.0037081, 0.0030206
7: -0.0214253, -0.0110046, -0.0211728, -0.0125405, -0.0054254, 0.0066898
8: 0.9624048, 0.9922614, 0.9631283, 0.9878607, -0.0192038, 0.0224671
9: -0.0000725, 0.0087024, 0.0012209, 0.0084898, -0.0058026, 0.0047687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116694, upper bound: 0.0109045
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116354, upper bound: 0.0108288
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004742, 0.0013369, -0.0004370, 0.0009720, -0.0011150, 0.0014403
1: -0.0008998, 0.0033706, -0.0007255, 0.0028113, -0.0029491, 0.0033062
2: 0.0112922, 0.0176875, 0.0121298, 0.0174265, -0.0046796, 0.0041109
3: -0.0021356, 0.0026734, -0.0015058, 0.0024771, -0.0033990, 0.0029626
4: -0.0063495, -0.0019137, -0.0057686, -0.0020947, -0.0041293, 0.0038549
5: 0.0058064, 0.0106067, 0.0064351, 0.0104108, -0.0033815, 0.0029458
6: 0.0067666, 0.0107417, 0.0076148, 0.0105045, -0.0037379, 0.0031269
7: -0.0214253, -0.0110046, -0.0210001, -0.0123694, -0.0055504, 0.0064477
8: 0.9624048, 0.9922614, 0.9636232, 0.9883512, -0.0195003, 0.0221226
9: -0.0000725, 0.0087024, 0.0010767, 0.0083443, -0.0056179, 0.0048662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116694, upper bound: 0.0109045
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116354, upper bound: 0.0108288
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004871, 0.0013133, -0.0004521, 0.0009263, -0.0010872, 0.0014381
1: -0.0009600, 0.0033343, -0.0007963, 0.0027411, -0.0029325, 0.0033115
2: 0.0113465, 0.0177776, 0.0122348, 0.0175325, -0.0047043, 0.0041043
3: -0.0020948, 0.0027411, -0.0014268, 0.0025568, -0.0034243, 0.0029643
4: -0.0063119, -0.0018512, -0.0056957, -0.0020212, -0.0040879, 0.0038051
5: 0.0058471, 0.0106743, 0.0065139, 0.0104903, -0.0034078, 0.0029483
6: 0.0068215, 0.0107263, 0.0077211, 0.0104747, -0.0036532, 0.0030052
7: -0.0215721, -0.0110930, -0.0211728, -0.0125405, -0.0055794, 0.0066047
8: 0.9619842, 0.9920081, 0.9631283, 0.9878607, -0.0194467, 0.0222231
9: 0.0000019, 0.0088260, 0.0012209, 0.0084898, -0.0057309, 0.0049057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112786, upper bound: 0.0106472
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111206, upper bound: 0.0103882
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004871, 0.0013133, -0.0004370, 0.0009720, -0.0011284, 0.0014195
1: -0.0009600, 0.0033343, -0.0007255, 0.0028113, -0.0029797, 0.0032744
2: 0.0113465, 0.0177776, 0.0121298, 0.0174265, -0.0046320, 0.0041826
3: -0.0020948, 0.0027411, -0.0015058, 0.0024771, -0.0033631, 0.0030271
4: -0.0063119, -0.0018512, -0.0057686, -0.0020947, -0.0040963, 0.0038726
5: 0.0058471, 0.0106743, 0.0064351, 0.0104108, -0.0033457, 0.0030106
6: 0.0068215, 0.0107263, 0.0076148, 0.0105045, -0.0036830, 0.0031116
7: -0.0215721, -0.0110930, -0.0210001, -0.0123694, -0.0057009, 0.0063700
8: 0.9619842, 0.9920081, 0.9636232, 0.9883512, -0.0198101, 0.0219001
9: 0.0000019, 0.0088260, 0.0010767, 0.0083443, -0.0055525, 0.0049992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112786, upper bound: 0.0106472
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111206, upper bound: 0.0103882
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004742, 0.0013369, -0.0004639, 0.0009006, -0.0010511, 0.0014766
1: -0.0008998, 0.0033706, -0.0008517, 0.0027019, -0.0028878, 0.0033823
2: 0.0112922, 0.0176875, 0.0122936, 0.0176155, -0.0048265, 0.0040239
3: -0.0021356, 0.0026734, -0.0013826, 0.0026192, -0.0035267, 0.0028988
4: -0.0063495, -0.0019137, -0.0056549, -0.0019637, -0.0041172, 0.0037412
5: 0.0058064, 0.0106067, 0.0065581, 0.0105526, -0.0035111, 0.0028819
6: 0.0067666, 0.0107417, 0.0077807, 0.0104581, -0.0036915, 0.0029610
7: -0.0214253, -0.0110046, -0.0213079, -0.0126364, -0.0053853, 0.0068881
8: 0.9624048, 0.9922614, 0.9627411, 0.9875861, -0.0190891, 0.0227723
9: -0.0000725, 0.0087024, 0.0013016, 0.0086036, -0.0059702, 0.0047350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111892, upper bound: 0.0106120
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110651, upper bound: 0.0104351
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004742, 0.0013369, -0.0004493, 0.0009464, -0.0010951, 0.0014576
1: -0.0008998, 0.0033706, -0.0007832, 0.0027720, -0.0029326, 0.0033399
2: 0.0112922, 0.0176875, 0.0121886, 0.0175129, -0.0047447, 0.0040862
3: -0.0021356, 0.0026734, -0.0014616, 0.0025421, -0.0034571, 0.0029440
4: -0.0063495, -0.0019137, -0.0057278, -0.0020348, -0.0041185, 0.0038141
5: 0.0058064, 0.0106067, 0.0064792, 0.0104756, -0.0034403, 0.0029273
6: 0.0067666, 0.0107417, 0.0076743, 0.0104878, -0.0037212, 0.0030674
7: -0.0214253, -0.0110046, -0.0211408, -0.0124652, -0.0055102, 0.0066412
8: 0.9624048, 0.9922614, 0.9632200, 0.9880767, -0.0193850, 0.0224141
9: -0.0000725, 0.0087024, 0.0011574, 0.0084628, -0.0057867, 0.0048323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111892, upper bound: 0.0106120
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110651, upper bound: 0.0104351
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004871, 0.0013133, -0.0004639, 0.0009006, -0.0010594, 0.0014477
1: -0.0009600, 0.0033343, -0.0008517, 0.0027019, -0.0028864, 0.0033150
2: 0.0113465, 0.0177776, 0.0122936, 0.0176155, -0.0047201, 0.0040314
3: -0.0020948, 0.0027411, -0.0013826, 0.0026192, -0.0034436, 0.0029105
4: -0.0063119, -0.0018512, -0.0056549, -0.0019637, -0.0040680, 0.0037701
5: 0.0058471, 0.0106743, 0.0065581, 0.0105526, -0.0034278, 0.0028940
6: 0.0068215, 0.0107263, 0.0077807, 0.0104581, -0.0036366, 0.0029456
7: -0.0215721, -0.0110930, -0.0213079, -0.0126364, -0.0054507, 0.0067163
8: 0.9619842, 0.9920081, 0.9627411, 0.9875861, -0.0191100, 0.0222847
9: 0.0000019, 0.0088260, 0.0013016, 0.0086036, -0.0058096, 0.0047849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111675, upper bound: 0.0106038
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109915, upper bound: 0.0103454
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004871, 0.0013133, -0.0004493, 0.0009464, -0.0011017, 0.0014289
1: -0.0009600, 0.0033343, -0.0007832, 0.0027720, -0.0029301, 0.0032741
2: 0.0113465, 0.0177776, 0.0121886, 0.0175129, -0.0046432, 0.0040870
3: -0.0020948, 0.0027411, -0.0014616, 0.0025421, -0.0033774, 0.0029502
4: -0.0063119, -0.0018512, -0.0057278, -0.0020348, -0.0040714, 0.0038337
5: 0.0058471, 0.0106743, 0.0064792, 0.0104756, -0.0033605, 0.0029348
6: 0.0068215, 0.0107263, 0.0076743, 0.0104878, -0.0036663, 0.0030520
7: -0.0215721, -0.0110930, -0.0211408, -0.0124652, -0.0055748, 0.0064729
8: 0.9619842, 0.9920081, 0.9632200, 0.9880767, -0.0193814, 0.0219418
9: 0.0000019, 0.0088260, 0.0011574, 0.0084628, -0.0056229, 0.0048816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111675, upper bound: 0.0106038
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109915, upper bound: 0.0103454
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004879, 0.0012858, -0.0004879, 0.0012858, -0.0013413, 0.0013413
1: -0.0009639, 0.0032921, -0.0009639, 0.0032921, -0.0032469, 0.0032469
2: 0.0114097, 0.0177836, 0.0114097, 0.0177836, -0.0044776, 0.0044776
3: -0.0020473, 0.0027456, -0.0020473, 0.0027456, -0.0032015, 0.0032015
4: -0.0062681, -0.0018471, -0.0062681, -0.0018471, -0.0043188, 0.0043188
5: 0.0058946, 0.0106788, 0.0058946, 0.0106788, -0.0031803, 0.0031803
6: 0.0068855, 0.0107084, 0.0068855, 0.0107084, -0.0038229, 0.0038229
7: -0.0215819, -0.0111960, -0.0215819, -0.0111960, -0.0056797, 0.0056797
8: 0.9619563, 0.9917130, 0.9619563, 0.9917130, -0.0212956, 0.0212956
9: 0.0000887, 0.0088343, 0.0000887, 0.0088343, -0.0050180, 0.0050180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119155, upper bound: 0.0110759
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118983, upper bound: 0.0110625
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004879, 0.0012858, -0.0005002, 0.0012606, -0.0013199, 0.0013580
1: -0.0009639, 0.0032921, -0.0010216, 0.0032535, -0.0032227, 0.0032674
2: 0.0114097, 0.0177836, 0.0114675, 0.0178700, -0.0045382, 0.0044413
3: -0.0020473, 0.0027456, -0.0020039, 0.0028106, -0.0032575, 0.0031743
4: -0.0062681, -0.0018471, -0.0062280, -0.0017872, -0.0043051, 0.0042937
5: 0.0058946, 0.0106788, 0.0059380, 0.0107436, -0.0032369, 0.0031531
6: 0.0068855, 0.0107084, 0.0069440, 0.0106921, -0.0038065, 0.0037644
7: -0.0215819, -0.0111960, -0.0217226, -0.0112902, -0.0056207, 0.0058717
8: 0.9619563, 0.9917130, 0.9615530, 0.9914432, -0.0211265, 0.0215417
9: 0.0000887, 0.0088343, 0.0001680, 0.0089527, -0.0051856, 0.0049683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119155, upper bound: 0.0110759
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118983, upper bound: 0.0110625
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0005002, 0.0012606, -0.0004879, 0.0012858, -0.0013580, 0.0013199
1: -0.0010216, 0.0032535, -0.0009639, 0.0032921, -0.0032674, 0.0032227
2: 0.0114675, 0.0178700, 0.0114097, 0.0177836, -0.0044413, 0.0045382
3: -0.0020039, 0.0028106, -0.0020473, 0.0027456, -0.0031743, 0.0032575
4: -0.0062280, -0.0017872, -0.0062681, -0.0018471, -0.0042937, 0.0043051
5: 0.0059380, 0.0107436, 0.0058946, 0.0106788, -0.0031531, 0.0032369
6: 0.0069440, 0.0106921, 0.0068855, 0.0107084, -0.0037644, 0.0038065
7: -0.0217226, -0.0112902, -0.0215819, -0.0111960, -0.0058717, 0.0056207
8: 0.9615530, 0.9914432, 0.9619563, 0.9917130, -0.0215417, 0.0211265
9: 0.0001680, 0.0089527, 0.0000887, 0.0088343, -0.0049683, 0.0051856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119069, upper bound: 0.0110713
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118785, upper bound: 0.0110449
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0005002, 0.0012606, -0.0005002, 0.0012606, -0.0013287, 0.0013287
1: -0.0010216, 0.0032535, -0.0010216, 0.0032535, -0.0032139, 0.0032139
2: 0.0114675, 0.0178700, 0.0114675, 0.0178700, -0.0044407, 0.0044407
3: -0.0020039, 0.0028106, -0.0020039, 0.0028106, -0.0031779, 0.0031779
4: -0.0062280, -0.0017872, -0.0062280, -0.0017872, -0.0042607, 0.0042607
5: 0.0059380, 0.0107436, 0.0059380, 0.0107436, -0.0031568, 0.0031568
6: 0.0069440, 0.0106921, 0.0069440, 0.0106921, -0.0037480, 0.0037480
7: -0.0217226, -0.0112902, -0.0217226, -0.0112902, -0.0057086, 0.0057086
8: 0.9615530, 0.9914432, 0.9615530, 0.9914432, -0.0211073, 0.0211073
9: 0.0001680, 0.0089527, 0.0001680, 0.0089527, -0.0050268, 0.0050268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119069, upper bound: 0.0110713
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118785, upper bound: 0.0110449
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004879, 0.0012858, -0.0004742, 0.0013369, -0.0013985, 0.0013365
1: -0.0009639, 0.0032921, -0.0008998, 0.0033706, -0.0033681, 0.0032895
2: 0.0114097, 0.0177836, 0.0112922, 0.0176875, -0.0045379, 0.0046590
3: -0.0020473, 0.0027456, -0.0021356, 0.0026734, -0.0032430, 0.0033379
4: -0.0062681, -0.0018471, -0.0063495, -0.0019137, -0.0043544, 0.0044447
5: 0.0058946, 0.0106788, 0.0058064, 0.0106067, -0.0032212, 0.0033165
6: 0.0068855, 0.0107084, 0.0067666, 0.0107417, -0.0038562, 0.0039419
7: -0.0215819, -0.0111960, -0.0214253, -0.0110046, -0.0059754, 0.0056508
8: 0.9619563, 0.9917130, 0.9624048, 0.9922614, -0.0221427, 0.0215805
9: 0.0000887, 0.0088343, -0.0000725, 0.0087024, -0.0050612, 0.0052670

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111621, upper bound: 0.0106309
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110678, upper bound: 0.0105027
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004879, 0.0012858, -0.0004871, 0.0013133, -0.0013787, 0.0013534
1: -0.0009639, 0.0032921, -0.0009600, 0.0033343, -0.0033466, 0.0033075
2: 0.0114097, 0.0177836, 0.0113465, 0.0177776, -0.0045851, 0.0046269
3: -0.0020473, 0.0027456, -0.0020948, 0.0027411, -0.0032881, 0.0033138
4: -0.0062681, -0.0018471, -0.0063119, -0.0018512, -0.0043802, 0.0044224
5: 0.0058946, 0.0106788, 0.0058471, 0.0106743, -0.0032671, 0.0032924
6: 0.0068855, 0.0107084, 0.0068215, 0.0107263, -0.0038408, 0.0038869
7: -0.0215819, -0.0111960, -0.0215721, -0.0110930, -0.0059230, 0.0058423
8: 0.9619563, 0.9917130, 0.9619842, 0.9920081, -0.0219926, 0.0217760
9: 0.0000887, 0.0088343, 0.0000019, 0.0088260, -0.0051915, 0.0052229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111621, upper bound: 0.0106309
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110678, upper bound: 0.0105027
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0005002, 0.0012606, -0.0004742, 0.0013369, -0.0014153, 0.0013151
1: -0.0010216, 0.0032535, -0.0008998, 0.0033706, -0.0033885, 0.0032653
2: 0.0114675, 0.0178700, 0.0112922, 0.0176875, -0.0045017, 0.0047196
3: -0.0020039, 0.0028106, -0.0021356, 0.0026734, -0.0032157, 0.0033939
4: -0.0062280, -0.0017872, -0.0063495, -0.0019137, -0.0043143, 0.0044310
5: 0.0059380, 0.0107436, 0.0058064, 0.0106067, -0.0031940, 0.0033731
6: 0.0069440, 0.0106921, 0.0067666, 0.0107417, -0.0037977, 0.0039255
7: -0.0217226, -0.0112902, -0.0214253, -0.0110046, -0.0061674, 0.0055918
8: 0.9615530, 0.9914432, 0.9624048, 0.9922614, -0.0223889, 0.0214114
9: 0.0001680, 0.0089527, -0.0000725, 0.0087024, -0.0050115, 0.0054346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111531, upper bound: 0.0106186
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110201, upper bound: 0.0104464
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0005002, 0.0012606, -0.0004871, 0.0013133, -0.0013876, 0.0013240
1: -0.0010216, 0.0032535, -0.0009600, 0.0033343, -0.0033363, 0.0032594
2: 0.0114675, 0.0178700, 0.0113465, 0.0177776, -0.0045052, 0.0046240
3: -0.0020039, 0.0028106, -0.0020948, 0.0027411, -0.0032232, 0.0033157
4: -0.0062280, -0.0017872, -0.0063119, -0.0018512, -0.0043329, 0.0043878
5: 0.0059380, 0.0107436, 0.0058471, 0.0106743, -0.0032018, 0.0032943
6: 0.0069440, 0.0106921, 0.0068215, 0.0107263, -0.0037823, 0.0038706
7: -0.0217226, -0.0112902, -0.0215721, -0.0110930, -0.0060073, 0.0056784
8: 0.9615530, 0.9914432, 0.9619842, 0.9920081, -0.0219630, 0.0214145
9: 0.0001680, 0.0089527, 0.0000019, 0.0088260, -0.0050651, 0.0052783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111531, upper bound: 0.0106186
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110201, upper bound: 0.0104464
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004742, 0.0013369, -0.0004879, 0.0012858, -0.0013365, 0.0013985
1: -0.0008998, 0.0033706, -0.0009639, 0.0032921, -0.0032895, 0.0033681
2: 0.0112922, 0.0176875, 0.0114097, 0.0177836, -0.0046590, 0.0045379
3: -0.0021356, 0.0026734, -0.0020473, 0.0027456, -0.0033379, 0.0032430
4: -0.0063495, -0.0019137, -0.0062681, -0.0018471, -0.0044447, 0.0043544
5: 0.0058064, 0.0106067, 0.0058946, 0.0106788, -0.0033165, 0.0032212
6: 0.0067666, 0.0107417, 0.0068855, 0.0107084, -0.0039419, 0.0038562
7: -0.0214253, -0.0110046, -0.0215819, -0.0111960, -0.0056508, 0.0059754
8: 0.9624048, 0.9922614, 0.9619563, 0.9917130, -0.0215805, 0.0221427
9: -0.0000725, 0.0087024, 0.0000887, 0.0088343, -0.0052670, 0.0050612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116469, upper bound: 0.0109045
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115772, upper bound: 0.0108288
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004742, 0.0013369, -0.0004742, 0.0013369, -0.0013800, 0.0013800
1: -0.0008998, 0.0033706, -0.0008998, 0.0033706, -0.0033451, 0.0033451
2: 0.0112922, 0.0176875, 0.0112922, 0.0176875, -0.0046005, 0.0046005
3: -0.0021356, 0.0026734, -0.0021356, 0.0026734, -0.0032817, 0.0032817
4: -0.0063495, -0.0019137, -0.0063495, -0.0019137, -0.0044358, 0.0044358
5: 0.0058064, 0.0106067, 0.0058064, 0.0106067, -0.0032592, 0.0032592
6: 0.0067666, 0.0107417, 0.0067666, 0.0107417, -0.0039751, 0.0039751
7: -0.0214253, -0.0110046, -0.0214253, -0.0110046, -0.0057361, 0.0057361
8: 0.9624048, 0.9922614, 0.9624048, 0.9922614, -0.0218916, 0.0218916
9: -0.0000725, 0.0087024, -0.0000725, 0.0087024, -0.0050891, 0.0050891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116469, upper bound: 0.0109045
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115772, upper bound: 0.0108288
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004871, 0.0013133, -0.0004879, 0.0012858, -0.0013534, 0.0013787
1: -0.0009600, 0.0033343, -0.0009639, 0.0032921, -0.0033075, 0.0033466
2: 0.0113465, 0.0177776, 0.0114097, 0.0177836, -0.0046269, 0.0045851
3: -0.0020948, 0.0027411, -0.0020473, 0.0027456, -0.0033138, 0.0032881
4: -0.0063119, -0.0018512, -0.0062681, -0.0018471, -0.0044224, 0.0043802
5: 0.0058471, 0.0106743, 0.0058946, 0.0106788, -0.0032924, 0.0032671
6: 0.0068215, 0.0107263, 0.0068855, 0.0107084, -0.0038869, 0.0038408
7: -0.0215721, -0.0110930, -0.0215819, -0.0111960, -0.0058423, 0.0059230
8: 0.9619842, 0.9920081, 0.9619563, 0.9917130, -0.0217760, 0.0219926
9: 0.0000019, 0.0088260, 0.0000887, 0.0088343, -0.0052229, 0.0051915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111993, upper bound: 0.0106472
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109626, upper bound: 0.0103882
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004871, 0.0013133, -0.0004742, 0.0013369, -0.0013963, 0.0013607
1: -0.0009600, 0.0033343, -0.0008998, 0.0033706, -0.0033642, 0.0033208
2: 0.0113465, 0.0177776, 0.0112922, 0.0176875, -0.0045641, 0.0046530
3: -0.0020948, 0.0027411, -0.0021356, 0.0026734, -0.0032543, 0.0033362
4: -0.0063119, -0.0018512, -0.0063495, -0.0019137, -0.0043982, 0.0044493
5: 0.0058471, 0.0106743, 0.0058064, 0.0106067, -0.0032319, 0.0033147
6: 0.0068215, 0.0107263, 0.0067666, 0.0107417, -0.0039202, 0.0039598
7: -0.0215721, -0.0110930, -0.0214253, -0.0110046, -0.0059255, 0.0056768
8: 0.9619842, 0.9920081, 0.9624048, 0.9922614, -0.0221099, 0.0217216
9: 0.0000019, 0.0088260, -0.0000725, 0.0087024, -0.0050391, 0.0052565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111993, upper bound: 0.0106472
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109626, upper bound: 0.0103882
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004742, 0.0013369, -0.0005002, 0.0012606, -0.0013151, 0.0014153
1: -0.0008998, 0.0033706, -0.0010216, 0.0032535, -0.0032653, 0.0033885
2: 0.0112922, 0.0176875, 0.0114675, 0.0178700, -0.0047196, 0.0045017
3: -0.0021356, 0.0026734, -0.0020039, 0.0028106, -0.0033939, 0.0032157
4: -0.0063495, -0.0019137, -0.0062280, -0.0017872, -0.0044310, 0.0043143
5: 0.0058064, 0.0106067, 0.0059380, 0.0107436, -0.0033731, 0.0031940
6: 0.0067666, 0.0107417, 0.0069440, 0.0106921, -0.0039255, 0.0037977
7: -0.0214253, -0.0110046, -0.0217226, -0.0112902, -0.0055918, 0.0061674
8: 0.9624048, 0.9922614, 0.9615530, 0.9914432, -0.0214114, 0.0223889
9: -0.0000725, 0.0087024, 0.0001680, 0.0089527, -0.0054346, 0.0050115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110837, upper bound: 0.0106120
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108916, upper bound: 0.0104351
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004742, 0.0013369, -0.0004871, 0.0013133, -0.0013607, 0.0013963
1: -0.0008998, 0.0033706, -0.0009600, 0.0033343, -0.0033208, 0.0033642
2: 0.0112922, 0.0176875, 0.0113465, 0.0177776, -0.0046530, 0.0045641
3: -0.0021356, 0.0026734, -0.0020948, 0.0027411, -0.0033362, 0.0032543
4: -0.0063495, -0.0019137, -0.0063119, -0.0018512, -0.0044493, 0.0043982
5: 0.0058064, 0.0106067, 0.0058471, 0.0106743, -0.0033147, 0.0032319
6: 0.0067666, 0.0107417, 0.0068215, 0.0107263, -0.0039598, 0.0039202
7: -0.0214253, -0.0110046, -0.0215721, -0.0110930, -0.0056768, 0.0059255
8: 0.9624048, 0.9922614, 0.9619842, 0.9920081, -0.0217216, 0.0221099
9: -0.0000725, 0.0087024, 0.0000019, 0.0088260, -0.0052565, 0.0050391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110837, upper bound: 0.0106120
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108916, upper bound: 0.0104351
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004871, 0.0013133, -0.0005002, 0.0012606, -0.0013240, 0.0013876
1: -0.0009600, 0.0033343, -0.0010216, 0.0032535, -0.0032594, 0.0033363
2: 0.0113465, 0.0177776, 0.0114675, 0.0178700, -0.0046240, 0.0045052
3: -0.0020948, 0.0027411, -0.0020039, 0.0028106, -0.0033157, 0.0032232
4: -0.0063119, -0.0018512, -0.0062280, -0.0017872, -0.0043878, 0.0043329
5: 0.0058471, 0.0106743, 0.0059380, 0.0107436, -0.0032943, 0.0032018
6: 0.0068215, 0.0107263, 0.0069440, 0.0106921, -0.0038706, 0.0037823
7: -0.0215721, -0.0110930, -0.0217226, -0.0112902, -0.0056784, 0.0060073
8: 0.9619842, 0.9920081, 0.9615530, 0.9914432, -0.0214145, 0.0219630
9: 0.0000019, 0.0088260, 0.0001680, 0.0089527, -0.0052783, 0.0050651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110577, upper bound: 0.0106038
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107812, upper bound: 0.0103454
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004871, 0.0013133, -0.0004871, 0.0013133, -0.0013692, 0.0013692
1: -0.0009600, 0.0033343, -0.0009600, 0.0033343, -0.0033129, 0.0033129
2: 0.0113465, 0.0177776, 0.0113465, 0.0177776, -0.0045629, 0.0045629
3: -0.0020948, 0.0027411, -0.0020948, 0.0027411, -0.0032593, 0.0032593
4: -0.0063119, -0.0018512, -0.0063119, -0.0018512, -0.0044081, 0.0044081
5: 0.0058471, 0.0106743, 0.0058471, 0.0106743, -0.0032375, 0.0032375
6: 0.0068215, 0.0107263, 0.0068215, 0.0107263, -0.0039048, 0.0039048
7: -0.0215721, -0.0110930, -0.0215721, -0.0110930, -0.0057651, 0.0057651
8: 0.9619842, 0.9920081, 0.9619842, 0.9920081, -0.0217075, 0.0217075
9: 0.0000019, 0.0088260, 0.0000019, 0.0088260, -0.0050946, 0.0050946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110577, upper bound: 0.0106038
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107812, upper bound: 0.0103454
time: 0.69 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.29 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0116466, upper bound: 0.0117010
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0116570, upper bound: 0.0116678
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0116466, upper bound: 0.0117010
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0116570, upper bound: 0.0116678
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0116466, upper bound: 0.0116860
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0116570, upper bound: 0.0116570
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0116466, upper bound: 0.0116860
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0116570, upper bound: 0.0116570
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0115797, upper bound: 0.0116308
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0115917, upper bound: 0.0116022
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0115797, upper bound: 0.0116308
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0115917, upper bound: 0.0116022
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0115797, upper bound: 0.0116121
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0115917, upper bound: 0.0116010
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0115797, upper bound: 0.0116121
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0115917, upper bound: 0.0116010
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112693, upper bound: 0.0114289
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112581, upper bound: 0.0112581
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112693, upper bound: 0.0114289
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112581, upper bound: 0.0112581
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0116008, upper bound: 0.0116058
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0116022, upper bound: 0.0115806
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0116008, upper bound: 0.0116058
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0116022, upper bound: 0.0115806
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0109072, upper bound: 0.0111352
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0108064, upper bound: 0.0108669
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0109072, upper bound: 0.0111352
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0108064, upper bound: 0.0108669
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0115987, upper bound: 0.0116058
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0115998, upper bound: 0.0115806
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0115987, upper bound: 0.0116058
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0115998, upper bound: 0.0115806
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112631, upper bound: 0.0125512
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112738, upper bound: 0.0125316
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112631, upper bound: 0.0125512
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112738, upper bound: 0.0125316
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112616, upper bound: 0.0125321
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112721, upper bound: 0.0125114
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112616, upper bound: 0.0125321
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112721, upper bound: 0.0125114
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111991, upper bound: 0.0124227
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112066, upper bound: 0.0123993
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111991, upper bound: 0.0124227
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112066, upper bound: 0.0123993
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111983, upper bound: 0.0124003
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112051, upper bound: 0.0123868
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111983, upper bound: 0.0124003
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112051, upper bound: 0.0123868
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0109319, upper bound: 0.0119012
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0108288, upper bound: 0.0116354
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0109319, upper bound: 0.0119012
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0108288, upper bound: 0.0116354
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111887, upper bound: 0.0123973
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111889, upper bound: 0.0123751
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111887, upper bound: 0.0123973
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111889, upper bound: 0.0123751
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0105785, upper bound: 0.0114709
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0103882, upper bound: 0.0111206
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0105785, upper bound: 0.0114709
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0103882, upper bound: 0.0111206
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111885, upper bound: 0.0123973
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111886, upper bound: 0.0123751
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111885, upper bound: 0.0123973
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111886, upper bound: 0.0123751
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0119233, upper bound: 0.0110759
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0119101, upper bound: 0.0110625
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0119233, upper bound: 0.0110759
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0119101, upper bound: 0.0110625
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0119125, upper bound: 0.0110713
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0118938, upper bound: 0.0110449
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0119125, upper bound: 0.0110713
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0118938, upper bound: 0.0110449
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112676, upper bound: 0.0106309
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112022, upper bound: 0.0105027
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112676, upper bound: 0.0106309
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112022, upper bound: 0.0105027
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112610, upper bound: 0.0106186
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111652, upper bound: 0.0104464
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112610, upper bound: 0.0106186
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111652, upper bound: 0.0104464
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0116694, upper bound: 0.0109045
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0116354, upper bound: 0.0108288
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0116694, upper bound: 0.0109045
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0116354, upper bound: 0.0108288
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112786, upper bound: 0.0106472
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111206, upper bound: 0.0103882
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0112786, upper bound: 0.0106472
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111206, upper bound: 0.0103882
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111892, upper bound: 0.0106120
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0110651, upper bound: 0.0104351
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111892, upper bound: 0.0106120
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0110651, upper bound: 0.0104351
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111675, upper bound: 0.0106038
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0109915, upper bound: 0.0103454
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111675, upper bound: 0.0106038
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0109915, upper bound: 0.0103454
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0119155, upper bound: 0.0110759
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0118983, upper bound: 0.0110625
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0119155, upper bound: 0.0110759
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0118983, upper bound: 0.0110625
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0119069, upper bound: 0.0110713
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0118785, upper bound: 0.0110449
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0119069, upper bound: 0.0110713
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0118785, upper bound: 0.0110449
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111621, upper bound: 0.0106309
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0110678, upper bound: 0.0105027
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111621, upper bound: 0.0106309
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0110678, upper bound: 0.0105027
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111531, upper bound: 0.0106186
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0110201, upper bound: 0.0104464
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111531, upper bound: 0.0106186
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0110201, upper bound: 0.0104464
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0116469, upper bound: 0.0109045
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0115772, upper bound: 0.0108288
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0116469, upper bound: 0.0109045
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0115772, upper bound: 0.0108288
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111993, upper bound: 0.0106472
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0109626, upper bound: 0.0103882
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0111993, upper bound: 0.0106472
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0109626, upper bound: 0.0103882
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0110837, upper bound: 0.0106120
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0108916, upper bound: 0.0104351
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0110837, upper bound: 0.0106120
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0108916, upper bound: 0.0104351
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0110577, upper bound: 0.0106038
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0107812, upper bound: 0.0103454
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0110577, upper bound: 0.0106038
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 8, lower bound: -0.0107812, upper bound: 0.0103454

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004561, 0.0008986, -0.0004518, 0.0009180, -0.0010155, 0.0009963
1: -0.0008151, 0.0026987, -0.0007948, 0.0027285, -0.0025199, 0.0025234
2: 0.0122983, 0.0175608, 0.0122538, 0.0175303, -0.0035262, 0.0035247
3: -0.0013791, 0.0025780, -0.0014126, 0.0025551, -0.0025401, 0.0025421
4: -0.0056517, -0.0020016, -0.0056826, -0.0020228, -0.0032620, 0.0032476
5: 0.0065616, 0.0105115, 0.0065282, 0.0104886, -0.0025254, 0.0025276
6: 0.0077854, 0.0104567, 0.0077403, 0.0104693, -0.0026839, 0.0027164
7: -0.0212188, -0.0126440, -0.0211691, -0.0125715, -0.0047537, 0.0047050
8: 0.9629965, 0.9875644, 0.9631389, 0.9877722, -0.0167086, 0.0167219
9: 0.0013080, 0.0085285, 0.0012469, 0.0084867, -0.0041277, 0.0041601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114841, upper bound: 0.0114750
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114631, upper bound: 0.0115244
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004514, 0.0008960, -0.0004519, 0.0009177, -0.0010176, 0.0009941
1: -0.0007929, 0.0026948, -0.0007954, 0.0027280, -0.0025411, 0.0025133
2: 0.0123043, 0.0175274, 0.0122545, 0.0175311, -0.0035087, 0.0035576
3: -0.0013746, 0.0025530, -0.0014120, 0.0025558, -0.0025266, 0.0025675
4: -0.0056476, -0.0020248, -0.0056821, -0.0020222, -0.0032564, 0.0032638
5: 0.0065661, 0.0104865, 0.0065287, 0.0104893, -0.0025118, 0.0025532
6: 0.0077915, 0.0104550, 0.0077411, 0.0104691, -0.0026777, 0.0027140
7: -0.0211644, -0.0126537, -0.0211705, -0.0125726, -0.0047958, 0.0046740
8: 0.9631524, 0.9875365, 0.9631349, 0.9877690, -0.0168645, 0.0166396
9: 0.0013162, 0.0084827, 0.0012478, 0.0084878, -0.0040997, 0.0041982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114903, upper bound: 0.0114280
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114677, upper bound: 0.0114677
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004561, 0.0008986, -0.0004636, 0.0008918, -0.0009926, 0.0010141
1: -0.0008151, 0.0026987, -0.0008502, 0.0026883, -0.0025037, 0.0025587
2: 0.0122983, 0.0175608, 0.0123139, 0.0176132, -0.0035960, 0.0035004
3: -0.0013791, 0.0025780, -0.0013673, 0.0026175, -0.0026036, 0.0025238
4: -0.0056517, -0.0020016, -0.0056409, -0.0019653, -0.0032545, 0.0032307
5: 0.0065616, 0.0105115, 0.0065733, 0.0105509, -0.0025898, 0.0025093
6: 0.0077854, 0.0104567, 0.0078012, 0.0104523, -0.0026669, 0.0026555
7: -0.0212188, -0.0126440, -0.0213042, -0.0126694, -0.0047141, 0.0049033
8: 0.9629965, 0.9875644, 0.9627517, 0.9874915, -0.0165951, 0.0170260
9: 0.0013080, 0.0085285, 0.0013294, 0.0086004, -0.0042947, 0.0041267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113667, upper bound: 0.0113371
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112694, upper bound: 0.0113517
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004514, 0.0008960, -0.0004638, 0.0008922, -0.0009949, 0.0010110
1: -0.0007929, 0.0026948, -0.0008508, 0.0026890, -0.0025248, 0.0025475
2: 0.0123043, 0.0175274, 0.0123129, 0.0176142, -0.0035772, 0.0035333
3: -0.0013746, 0.0025530, -0.0013681, 0.0026182, -0.0025903, 0.0025492
4: -0.0056476, -0.0020248, -0.0056416, -0.0019646, -0.0032491, 0.0032469
5: 0.0065661, 0.0104865, 0.0065725, 0.0105516, -0.0025763, 0.0025349
6: 0.0077915, 0.0104550, 0.0078002, 0.0104526, -0.0026611, 0.0026549
7: -0.0211644, -0.0126537, -0.0213059, -0.0126677, -0.0047561, 0.0048712
8: 0.9631524, 0.9875365, 0.9627472, 0.9874965, -0.0167507, 0.0169363
9: 0.0013162, 0.0084827, 0.0013279, 0.0086018, -0.0042612, 0.0041648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113720, upper bound: 0.0112979
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112726, upper bound: 0.0113152
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004675, 0.0008712, -0.0004518, 0.0009180, -0.0010319, 0.0009735
1: -0.0008685, 0.0026568, -0.0007948, 0.0027285, -0.0025507, 0.0025081
2: 0.0123611, 0.0176407, 0.0122538, 0.0175303, -0.0035033, 0.0035926
3: -0.0013319, 0.0026382, -0.0014126, 0.0025551, -0.0025228, 0.0026048
4: -0.0056082, -0.0019462, -0.0056826, -0.0020228, -0.0032461, 0.0032390
5: 0.0066087, 0.0105715, 0.0065282, 0.0104886, -0.0025082, 0.0025909
6: 0.0078490, 0.0104390, 0.0077403, 0.0104693, -0.0026204, 0.0026986
7: -0.0213490, -0.0127462, -0.0211691, -0.0125715, -0.0049502, 0.0046676
8: 0.9626234, 0.9872714, 0.9631389, 0.9877722, -0.0169997, 0.0166149
9: 0.0013941, 0.0086382, 0.0012469, 0.0084867, -0.0040963, 0.0043125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113886, upper bound: 0.0113073
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113007, upper bound: 0.0113064
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004633, 0.0008703, -0.0004519, 0.0009177, -0.0010355, 0.0009712
1: -0.0008485, 0.0026554, -0.0007954, 0.0027280, -0.0025772, 0.0024968
2: 0.0123632, 0.0176107, 0.0122545, 0.0175311, -0.0034841, 0.0036299
3: -0.0013303, 0.0026156, -0.0014120, 0.0025558, -0.0025081, 0.0026331
4: -0.0056067, -0.0019670, -0.0056821, -0.0020222, -0.0032393, 0.0032595
5: 0.0066103, 0.0105490, 0.0065287, 0.0104893, -0.0024933, 0.0026196
6: 0.0078512, 0.0104384, 0.0077411, 0.0104691, -0.0026180, 0.0026973
7: -0.0213001, -0.0127497, -0.0211705, -0.0125726, -0.0049950, 0.0046339
8: 0.9627634, 0.9872614, 0.9631349, 0.9877690, -0.0171801, 0.0165246
9: 0.0013970, 0.0085970, 0.0012478, 0.0084878, -0.0040659, 0.0043657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114036, upper bound: 0.0112736
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113152, upper bound: 0.0112726
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004675, 0.0008712, -0.0004636, 0.0008918, -0.0010011, 0.0009821
1: -0.0008685, 0.0026568, -0.0008502, 0.0026883, -0.0024867, 0.0024932
2: 0.0123611, 0.0176407, 0.0123139, 0.0176132, -0.0034922, 0.0034867
3: -0.0013319, 0.0026382, -0.0013673, 0.0026175, -0.0025221, 0.0025215
4: -0.0056082, -0.0019462, -0.0056409, -0.0019653, -0.0032069, 0.0031869
5: 0.0066087, 0.0105715, 0.0065733, 0.0105509, -0.0025082, 0.0025079
6: 0.0078490, 0.0104390, 0.0078012, 0.0104523, -0.0026033, 0.0026377
7: -0.0213490, -0.0127462, -0.0213042, -0.0126694, -0.0047823, 0.0047365
8: 0.9626234, 0.9872714, 0.9627517, 0.9874915, -0.0165163, 0.0165481
9: 0.0013941, 0.0086382, 0.0013294, 0.0086004, -0.0041390, 0.0041675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113667, upper bound: 0.0113051
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112661, upper bound: 0.0113052
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004633, 0.0008703, -0.0004638, 0.0008922, -0.0010041, 0.0009802
1: -0.0008485, 0.0026554, -0.0008508, 0.0026890, -0.0025102, 0.0024838
2: 0.0123632, 0.0176107, 0.0123129, 0.0176142, -0.0034742, 0.0035229
3: -0.0013303, 0.0026156, -0.0013681, 0.0026182, -0.0025072, 0.0025490
4: -0.0056067, -0.0019670, -0.0056416, -0.0019646, -0.0032002, 0.0032090
5: 0.0066103, 0.0105490, 0.0065725, 0.0105516, -0.0024933, 0.0025354
6: 0.0078512, 0.0104384, 0.0078002, 0.0104526, -0.0026014, 0.0026382
7: -0.0213001, -0.0127497, -0.0213059, -0.0126677, -0.0048242, 0.0047054
8: 0.9627634, 0.9872614, 0.9627472, 0.9874965, -0.0166863, 0.0164665
9: 0.0013970, 0.0085970, 0.0013279, 0.0086018, -0.0041100, 0.0042070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113720, upper bound: 0.0112715
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112714, upper bound: 0.0112714
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004561, 0.0008986, -0.0004367, 0.0009632, -0.0010670, 0.0009924
1: -0.0008151, 0.0026987, -0.0007241, 0.0027978, -0.0026504, 0.0025676
2: 0.0122983, 0.0175608, 0.0121500, 0.0174243, -0.0035860, 0.0037202
3: -0.0013791, 0.0025780, -0.0014906, 0.0024755, -0.0025841, 0.0026891
4: -0.0056517, -0.0020016, -0.0057546, -0.0020963, -0.0033292, 0.0033832
5: 0.0065616, 0.0105115, 0.0064503, 0.0104091, -0.0025693, 0.0026742
6: 0.0077854, 0.0104567, 0.0076353, 0.0104987, -0.0027133, 0.0028214
7: -0.0212188, -0.0126440, -0.0209965, -0.0124024, -0.0050721, 0.0046820
8: 0.9629965, 0.9875644, 0.9636333, 0.9882566, -0.0176210, 0.0170014
9: 0.0013080, 0.0085285, 0.0011045, 0.0083413, -0.0041544, 0.0044282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112247, upper bound: 0.0111038
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110379, upper bound: 0.0110678
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004514, 0.0008960, -0.0004368, 0.0009645, -0.0010707, 0.0009899
1: -0.0007929, 0.0026948, -0.0007247, 0.0027998, -0.0026711, 0.0025592
2: 0.0123043, 0.0175274, 0.0121469, 0.0174253, -0.0035714, 0.0037524
3: -0.0013746, 0.0025530, -0.0014929, 0.0024762, -0.0025712, 0.0027140
4: -0.0056476, -0.0020248, -0.0057567, -0.0020956, -0.0033256, 0.0033988
5: 0.0065661, 0.0104865, 0.0064480, 0.0104098, -0.0025561, 0.0026993
6: 0.0077915, 0.0104550, 0.0076321, 0.0104996, -0.0027081, 0.0028229
7: -0.0211644, -0.0126537, -0.0209980, -0.0123973, -0.0051131, 0.0046519
8: 0.9631524, 0.9875365, 0.9636291, 0.9882711, -0.0177735, 0.0169405
9: 0.0013162, 0.0084827, 0.0011003, 0.0083426, -0.0041310, 0.0044654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112293, upper bound: 0.0110661
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110401, upper bound: 0.0110423
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004561, 0.0008986, -0.0004490, 0.0009366, -0.0010468, 0.0010095
1: -0.0008151, 0.0026987, -0.0007817, 0.0027570, -0.0026299, 0.0025979
2: 0.0122983, 0.0175608, 0.0122111, 0.0175107, -0.0036388, 0.0036895
3: -0.0013791, 0.0025780, -0.0014447, 0.0025404, -0.0026274, 0.0026660
4: -0.0056517, -0.0020016, -0.0057122, -0.0020363, -0.0033184, 0.0033619
5: 0.0065616, 0.0105115, 0.0064961, 0.0104740, -0.0026127, 0.0026512
6: 0.0077854, 0.0104567, 0.0076971, 0.0104814, -0.0026960, 0.0027597
7: -0.0212188, -0.0126440, -0.0211373, -0.0125018, -0.0050221, 0.0048786
8: 0.9629965, 0.9875644, 0.9632300, 0.9879717, -0.0174777, 0.0172424
9: 0.0013080, 0.0085285, 0.0011883, 0.0084599, -0.0043017, 0.0043861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110255, upper bound: 0.0108966
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107613, upper bound: 0.0107967
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004514, 0.0008960, -0.0004491, 0.0009394, -0.0010515, 0.0010061
1: -0.0007929, 0.0026948, -0.0007824, 0.0027614, -0.0026534, 0.0025888
2: 0.0123043, 0.0175274, 0.0122045, 0.0175117, -0.0036245, 0.0037259
3: -0.0013746, 0.0025530, -0.0014496, 0.0025412, -0.0026151, 0.0026940
4: -0.0056476, -0.0020248, -0.0057168, -0.0020356, -0.0033169, 0.0033805
5: 0.0065661, 0.0104865, 0.0064912, 0.0104747, -0.0026000, 0.0026795
6: 0.0077915, 0.0104550, 0.0076905, 0.0104833, -0.0026918, 0.0027646
7: -0.0211644, -0.0126537, -0.0211389, -0.0124912, -0.0050699, 0.0048467
8: 0.9631524, 0.9875365, 0.9632255, 0.9880022, -0.0176498, 0.0171793
9: 0.0013162, 0.0084827, 0.0011793, 0.0084612, -0.0042778, 0.0044290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110283, upper bound: 0.0108573
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107630, upper bound: 0.0107621
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004675, 0.0008712, -0.0004367, 0.0009632, -0.0010834, 0.0009696
1: -0.0008685, 0.0026568, -0.0007241, 0.0027978, -0.0026812, 0.0025523
2: 0.0123611, 0.0176407, 0.0121500, 0.0174243, -0.0035631, 0.0037880
3: -0.0013319, 0.0026382, -0.0014906, 0.0024755, -0.0025669, 0.0027518
4: -0.0056082, -0.0019462, -0.0057546, -0.0020963, -0.0033133, 0.0033745
5: 0.0066087, 0.0105715, 0.0064503, 0.0104091, -0.0025521, 0.0027376
6: 0.0078490, 0.0104390, 0.0076353, 0.0104987, -0.0026497, 0.0028037
7: -0.0213490, -0.0127462, -0.0209965, -0.0124024, -0.0052687, 0.0046447
8: 0.9626234, 0.9872714, 0.9636333, 0.9882566, -0.0179121, 0.0168944
9: 0.0013941, 0.0086382, 0.0011045, 0.0083413, -0.0041229, 0.0045807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110469, upper bound: 0.0108581
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108260, upper bound: 0.0107281
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004633, 0.0008703, -0.0004368, 0.0009645, -0.0010885, 0.0009670
1: -0.0008485, 0.0026554, -0.0007247, 0.0027998, -0.0027072, 0.0025428
2: 0.0123632, 0.0176107, 0.0121469, 0.0174253, -0.0035468, 0.0038246
3: -0.0013303, 0.0026156, -0.0014929, 0.0024762, -0.0025527, 0.0027795
4: -0.0056067, -0.0019670, -0.0057567, -0.0020956, -0.0033085, 0.0033946
5: 0.0066103, 0.0105490, 0.0064480, 0.0104098, -0.0025376, 0.0027657
6: 0.0078512, 0.0104384, 0.0076321, 0.0104996, -0.0026484, 0.0028062
7: -0.0213001, -0.0127497, -0.0209980, -0.0123973, -0.0053123, 0.0046118
8: 0.9627634, 0.9872614, 0.9636291, 0.9882711, -0.0180891, 0.0168255
9: 0.0013970, 0.0085970, 0.0011003, 0.0083426, -0.0040972, 0.0046329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110531, upper bound: 0.0108390
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108355, upper bound: 0.0107182
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004675, 0.0008712, -0.0004490, 0.0009366, -0.0010539, 0.0009780
1: -0.0008685, 0.0026568, -0.0007817, 0.0027570, -0.0026169, 0.0025399
2: 0.0123611, 0.0176407, 0.0122111, 0.0175107, -0.0035581, 0.0036816
3: -0.0013319, 0.0026382, -0.0014447, 0.0025404, -0.0025694, 0.0026681
4: -0.0056082, -0.0019462, -0.0057122, -0.0020363, -0.0032763, 0.0033222
5: 0.0066087, 0.0105715, 0.0064961, 0.0104740, -0.0025551, 0.0026543
6: 0.0078490, 0.0104390, 0.0076971, 0.0104814, -0.0026325, 0.0027419
7: -0.0213490, -0.0127462, -0.0211373, -0.0125018, -0.0051000, 0.0047120
8: 0.9626234, 0.9872714, 0.9632300, 0.9879717, -0.0174266, 0.0168628
9: 0.0013941, 0.0086382, 0.0011883, 0.0084599, -0.0041593, 0.0044351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110178, upper bound: 0.0108396
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107399, upper bound: 0.0106936
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004633, 0.0008703, -0.0004491, 0.0009394, -0.0010592, 0.0009759
1: -0.0008485, 0.0026554, -0.0007824, 0.0027614, -0.0026404, 0.0025326
2: 0.0123632, 0.0176107, 0.0122045, 0.0175117, -0.0035432, 0.0037178
3: -0.0013303, 0.0026156, -0.0014496, 0.0025412, -0.0025560, 0.0026956
4: -0.0056067, -0.0019670, -0.0057168, -0.0020356, -0.0032739, 0.0033443
5: 0.0066103, 0.0105490, 0.0064912, 0.0104747, -0.0025415, 0.0026817
6: 0.0078512, 0.0104384, 0.0076905, 0.0104833, -0.0026321, 0.0027479
7: -0.0213001, -0.0127497, -0.0211389, -0.0124912, -0.0051418, 0.0046795
8: 0.9627634, 0.9872614, 0.9632255, 0.9880022, -0.0175964, 0.0167956
9: 0.0013970, 0.0085970, 0.0011793, 0.0084612, -0.0041362, 0.0044744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110216, upper bound: 0.0108186
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107439, upper bound: 0.0106861
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004304, 0.0009698, -0.0004521, 0.0009263, -0.0010128, 0.0010759
1: -0.0006945, 0.0028079, -0.0007963, 0.0027411, -0.0025859, 0.0027038
2: 0.0121348, 0.0173801, 0.0122348, 0.0175325, -0.0037940, 0.0036106
3: -0.0015020, 0.0024422, -0.0014268, 0.0025568, -0.0027398, 0.0026000
4: -0.0057651, -0.0021269, -0.0056957, -0.0020212, -0.0034566, 0.0033566
5: 0.0064389, 0.0103759, 0.0065139, 0.0104903, -0.0027246, 0.0025846
6: 0.0076199, 0.0105030, 0.0077211, 0.0104747, -0.0028548, 0.0027819
7: -0.0209244, -0.0123776, -0.0211728, -0.0125405, -0.0046999, 0.0051215
8: 0.9638399, 0.9883276, 0.9631283, 0.9878607, -0.0171224, 0.0179737
9: 0.0010836, 0.0082806, 0.0012209, 0.0084898, -0.0044820, 0.0041775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111038, upper bound: 0.0112247
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110661, upper bound: 0.0112293
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004199, 0.0010543, -0.0004469, 0.0009237, -0.0009978, 0.0011533
1: -0.0006456, 0.0029374, -0.0007719, 0.0027372, -0.0027202, 0.0027902
2: 0.0119409, 0.0173068, 0.0122407, 0.0174960, -0.0039270, 0.0037630
3: -0.0016479, 0.0023871, -0.0014224, 0.0025294, -0.0028448, 0.0026939
4: -0.0058996, -0.0021778, -0.0056917, -0.0020465, -0.0035403, 0.0035139
5: 0.0062933, 0.0103209, 0.0065184, 0.0104629, -0.0028297, 0.0026763
6: 0.0074235, 0.0105580, 0.0077271, 0.0104731, -0.0030496, 0.0028309
7: -0.0208050, -0.0120616, -0.0211133, -0.0125501, -0.0045860, 0.0053783
8: 0.9641820, 0.9892331, 0.9632988, 0.9878333, -0.0178909, 0.0185917
9: 0.0008176, 0.0081801, 0.0012289, 0.0084397, -0.0046971, 0.0041786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110678, upper bound: 0.0110379
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110423, upper bound: 0.0110401
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004304, 0.0009698, -0.0004370, 0.0009720, -0.0010501, 0.0010575
1: -0.0006945, 0.0028079, -0.0007255, 0.0028113, -0.0026206, 0.0026531
2: 0.0121348, 0.0173801, 0.0121298, 0.0174265, -0.0037016, 0.0036510
3: -0.0015020, 0.0024422, -0.0015058, 0.0024771, -0.0026635, 0.0026259
4: -0.0057651, -0.0021269, -0.0057686, -0.0020947, -0.0034509, 0.0034193
5: 0.0064389, 0.0103759, 0.0064351, 0.0104108, -0.0026473, 0.0026100
6: 0.0076199, 0.0105030, 0.0076148, 0.0105045, -0.0028846, 0.0028883
7: -0.0209244, -0.0123776, -0.0210001, -0.0123694, -0.0047639, 0.0048540
8: 0.9638399, 0.9883276, 0.9636232, 0.9883512, -0.0173237, 0.0175565
9: 0.0010836, 0.0082806, 0.0010767, 0.0083443, -0.0042759, 0.0041980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112581, upper bound: 0.0112581
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112581, upper bound: 0.0112581
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004199, 0.0010543, -0.0004318, 0.0009695, -0.0010330, 0.0011348
1: -0.0006456, 0.0029374, -0.0007014, 0.0028074, -0.0027585, 0.0027430
2: 0.0119409, 0.0173068, 0.0121356, 0.0173904, -0.0038386, 0.0038104
3: -0.0016479, 0.0023871, -0.0015014, 0.0024499, -0.0027686, 0.0027218
4: -0.0058996, -0.0021778, -0.0057646, -0.0021198, -0.0035362, 0.0035868
5: 0.0062933, 0.0103209, 0.0064395, 0.0103837, -0.0027525, 0.0027037
6: 0.0074235, 0.0105580, 0.0076207, 0.0105028, -0.0030793, 0.0029373
7: -0.0208050, -0.0120616, -0.0209412, -0.0123789, -0.0046184, 0.0051144
8: 0.9641820, 0.9892331, 0.9637918, 0.9883239, -0.0181211, 0.0181916
9: 0.0008176, 0.0081801, 0.0010847, 0.0082947, -0.0044944, 0.0041870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110478, upper bound: 0.0109863
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109863, upper bound: 0.0109863
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004495, 0.0009145, -0.0004518, 0.0009180, -0.0010269, 0.0010250
1: -0.0007843, 0.0027231, -0.0007948, 0.0027285, -0.0025910, 0.0026324
2: 0.0122619, 0.0175145, 0.0122538, 0.0175303, -0.0036895, 0.0036340
3: -0.0014065, 0.0025433, -0.0014126, 0.0025551, -0.0026628, 0.0026249
4: -0.0056770, -0.0020337, -0.0056826, -0.0020228, -0.0033752, 0.0033009
5: 0.0065342, 0.0104768, 0.0065282, 0.0104886, -0.0026480, 0.0026102
6: 0.0077485, 0.0104671, 0.0077403, 0.0104693, -0.0027208, 0.0027267
7: -0.0211434, -0.0125846, -0.0211691, -0.0125715, -0.0049117, 0.0049710
8: 0.9632125, 0.9877344, 0.9631389, 0.9877722, -0.0172151, 0.0174840
9: 0.0012580, 0.0084650, 0.0012469, 0.0084867, -0.0043517, 0.0043241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110287, upper bound: 0.0108882
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107621, upper bound: 0.0108168
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004487, 0.0009208, -0.0004519, 0.0009177, -0.0010309, 0.0010294
1: -0.0007803, 0.0027328, -0.0007954, 0.0027280, -0.0026220, 0.0026237
2: 0.0122473, 0.0175086, 0.0122545, 0.0175311, -0.0036740, 0.0036788
3: -0.0014174, 0.0025388, -0.0014120, 0.0025558, -0.0026509, 0.0026599
4: -0.0056871, -0.0020378, -0.0056821, -0.0020222, -0.0033711, 0.0033298
5: 0.0065233, 0.0104723, 0.0065287, 0.0104893, -0.0026358, 0.0026454
6: 0.0077338, 0.0104712, 0.0077411, 0.0104691, -0.0027353, 0.0027301
7: -0.0211338, -0.0125610, -0.0211705, -0.0125726, -0.0049709, 0.0049434
8: 0.9632401, 0.9878023, 0.9631349, 0.9877690, -0.0174243, 0.0174113
9: 0.0012380, 0.0084569, 0.0012478, 0.0084878, -0.0043265, 0.0043750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110287, upper bound: 0.0108400
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107621, upper bound: 0.0107630
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004495, 0.0009145, -0.0004367, 0.0009632, -0.0010647, 0.0010071
1: -0.0007843, 0.0027231, -0.0007241, 0.0027978, -0.0026293, 0.0025848
2: 0.0122619, 0.0175145, 0.0121500, 0.0174243, -0.0036021, 0.0036860
3: -0.0014065, 0.0025433, -0.0014906, 0.0024755, -0.0025897, 0.0026665
4: -0.0056770, -0.0020337, -0.0057546, -0.0020963, -0.0033719, 0.0033598
5: 0.0065342, 0.0104768, 0.0064503, 0.0104091, -0.0025739, 0.0026518
6: 0.0077485, 0.0104671, 0.0076353, 0.0104987, -0.0027502, 0.0028318
7: -0.0211434, -0.0125846, -0.0209965, -0.0124024, -0.0049878, 0.0047116
8: 0.9632125, 0.9877344, 0.9636333, 0.9882566, -0.0174601, 0.0170888
9: 0.0012580, 0.0084650, 0.0011045, 0.0083413, -0.0041520, 0.0043658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109910, upper bound: 0.0107993
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107068, upper bound: 0.0106643
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004487, 0.0009208, -0.0004368, 0.0009645, -0.0010696, 0.0010106
1: -0.0007803, 0.0027328, -0.0007247, 0.0027998, -0.0026589, 0.0025794
2: 0.0122473, 0.0175086, 0.0121469, 0.0174253, -0.0035902, 0.0037328
3: -0.0014174, 0.0025388, -0.0014929, 0.0024762, -0.0025801, 0.0026986
4: -0.0056871, -0.0020378, -0.0057567, -0.0020956, -0.0033724, 0.0033890
5: 0.0065233, 0.0104723, 0.0064480, 0.0104098, -0.0025643, 0.0026836
6: 0.0077338, 0.0104712, 0.0076321, 0.0104996, -0.0027658, 0.0028390
7: -0.0211338, -0.0125610, -0.0209980, -0.0123973, -0.0050394, 0.0046860
8: 0.9632401, 0.9878023, 0.9636291, 0.9882711, -0.0176753, 0.0170395
9: 0.0012380, 0.0084569, 0.0011003, 0.0083426, -0.0041292, 0.0044292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109910, upper bound: 0.0107699
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107068, upper bound: 0.0106472
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004304, 0.0009698, -0.0004639, 0.0009006, -0.0009899, 0.0010938
1: -0.0006945, 0.0028079, -0.0008517, 0.0027019, -0.0025694, 0.0027396
2: 0.0121348, 0.0173801, 0.0122936, 0.0176155, -0.0038640, 0.0035860
3: -0.0015020, 0.0024422, -0.0013826, 0.0026192, -0.0028029, 0.0025815
4: -0.0057651, -0.0021269, -0.0056549, -0.0019637, -0.0034496, 0.0033395
5: 0.0064389, 0.0103759, 0.0065581, 0.0105526, -0.0027887, 0.0025662
6: 0.0076199, 0.0105030, 0.0077807, 0.0104581, -0.0028382, 0.0027223
7: -0.0209244, -0.0123776, -0.0213079, -0.0126364, -0.0046598, 0.0053198
8: 0.9638399, 0.9883276, 0.9627411, 0.9875861, -0.0170076, 0.0182788
9: 0.0010836, 0.0082806, 0.0013016, 0.0086036, -0.0046496, 0.0041438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108581, upper bound: 0.0110469
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108390, upper bound: 0.0110531
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004199, 0.0010543, -0.0004590, 0.0008980, -0.0009738, 0.0011710
1: -0.0006456, 0.0029374, -0.0008284, 0.0026979, -0.0027008, 0.0028260
2: 0.0119409, 0.0173068, 0.0122996, 0.0175806, -0.0040051, 0.0037340
3: -0.0016479, 0.0023871, -0.0013781, 0.0025929, -0.0029128, 0.0026721
4: -0.0058996, -0.0021778, -0.0056508, -0.0019879, -0.0035351, 0.0034731
5: 0.0062933, 0.0103209, 0.0065625, 0.0105264, -0.0028985, 0.0026545
6: 0.0074235, 0.0105580, 0.0077867, 0.0104564, -0.0030329, 0.0027712
7: -0.0208050, -0.0120616, -0.0212510, -0.0126461, -0.0045387, 0.0055702
8: 0.9641820, 0.9892331, 0.9629042, 0.9875584, -0.0177554, 0.0189289
9: 0.0008176, 0.0081801, 0.0013097, 0.0085556, -0.0048576, 0.0041388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107281, upper bound: 0.0108260
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107182, upper bound: 0.0108355
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004304, 0.0009698, -0.0004493, 0.0009464, -0.0010301, 0.0010748
1: -0.0006945, 0.0028079, -0.0007832, 0.0027720, -0.0026041, 0.0026868
2: 0.0121348, 0.0173801, 0.0121886, 0.0175129, -0.0037666, 0.0036263
3: -0.0015020, 0.0024422, -0.0014616, 0.0025421, -0.0027217, 0.0026073
4: -0.0057651, -0.0021269, -0.0057278, -0.0020348, -0.0034401, 0.0034022
5: 0.0064389, 0.0103759, 0.0064792, 0.0104756, -0.0027062, 0.0025915
6: 0.0076199, 0.0105030, 0.0076743, 0.0104878, -0.0028679, 0.0028287
7: -0.0209244, -0.0123776, -0.0211408, -0.0124652, -0.0047236, 0.0050475
8: 0.9638399, 0.9883276, 0.9632200, 0.9880767, -0.0172084, 0.0178480
9: 0.0010836, 0.0082806, 0.0011574, 0.0084628, -0.0044447, 0.0041642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107993, upper bound: 0.0109910
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107699, upper bound: 0.0109910
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004199, 0.0010543, -0.0004439, 0.0009438, -0.0010127, 0.0011520
1: -0.0006456, 0.0029374, -0.0007580, 0.0027681, -0.0027419, 0.0027768
2: 0.0119409, 0.0173068, 0.0121945, 0.0174751, -0.0039094, 0.0037855
3: -0.0016479, 0.0023871, -0.0014571, 0.0025137, -0.0028324, 0.0027030
4: -0.0058996, -0.0021778, -0.0057237, -0.0020610, -0.0035267, 0.0035459
5: 0.0062933, 0.0103209, 0.0064837, 0.0104473, -0.0028171, 0.0026850
6: 0.0074235, 0.0105580, 0.0076803, 0.0104861, -0.0030627, 0.0028776
7: -0.0208050, -0.0120616, -0.0210793, -0.0124749, -0.0045778, 0.0053064
8: 0.9641820, 0.9892331, 0.9633963, 0.9880489, -0.0180048, 0.0184963
9: 0.0008176, 0.0081801, 0.0011656, 0.0084110, -0.0046562, 0.0041529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106643, upper bound: 0.0107068
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106472, upper bound: 0.0107068
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004495, 0.0009145, -0.0004636, 0.0008918, -0.0009962, 0.0010326
1: -0.0007843, 0.0027231, -0.0008502, 0.0026883, -0.0025354, 0.0026227
2: 0.0122619, 0.0175145, 0.0123139, 0.0176132, -0.0036860, 0.0035535
3: -0.0014065, 0.0025433, -0.0013673, 0.0026175, -0.0026679, 0.0025681
4: -0.0056770, -0.0020337, -0.0056409, -0.0019653, -0.0033413, 0.0032567
5: 0.0065342, 0.0104768, 0.0065733, 0.0105509, -0.0026537, 0.0025540
6: 0.0077485, 0.0104671, 0.0078012, 0.0104523, -0.0027038, 0.0026658
7: -0.0211434, -0.0125846, -0.0213042, -0.0126694, -0.0047499, 0.0050524
8: 0.9632125, 0.9877344, 0.9627517, 0.9874915, -0.0168368, 0.0174531
9: 0.0012580, 0.0084650, 0.0013294, 0.0086004, -0.0044050, 0.0041828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109779, upper bound: 0.0108756
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106932, upper bound: 0.0107885
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004487, 0.0009208, -0.0004638, 0.0008922, -0.0009999, 0.0010369
1: -0.0007803, 0.0027328, -0.0008508, 0.0026890, -0.0025626, 0.0026145
2: 0.0122473, 0.0175086, 0.0123129, 0.0176142, -0.0036701, 0.0035962
3: -0.0014174, 0.0025388, -0.0013681, 0.0026182, -0.0026545, 0.0026003
4: -0.0056871, -0.0020378, -0.0056416, -0.0019646, -0.0033361, 0.0032859
5: 0.0065233, 0.0104723, 0.0065725, 0.0105516, -0.0026403, 0.0025860
6: 0.0077338, 0.0104712, 0.0078002, 0.0104526, -0.0027188, 0.0026710
7: -0.0211338, -0.0125610, -0.0213059, -0.0126677, -0.0048002, 0.0050246
8: 0.9632401, 0.9878023, 0.9627472, 0.9874965, -0.0170356, 0.0173809
9: 0.0012380, 0.0084569, 0.0013279, 0.0086018, -0.0043787, 0.0042279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109766, upper bound: 0.0108309
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106861, upper bound: 0.0107439
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004495, 0.0009145, -0.0004490, 0.0009366, -0.0010344, 0.0010143
1: -0.0007843, 0.0027231, -0.0007817, 0.0027570, -0.0025662, 0.0025711
2: 0.0122619, 0.0175145, 0.0122111, 0.0175107, -0.0035933, 0.0035857
3: -0.0014065, 0.0025433, -0.0014447, 0.0025404, -0.0025892, 0.0025867
4: -0.0056770, -0.0020337, -0.0057122, -0.0020363, -0.0033328, 0.0033115
5: 0.0065342, 0.0104768, 0.0064961, 0.0104740, -0.0025738, 0.0025716
6: 0.0077485, 0.0104671, 0.0076971, 0.0104814, -0.0027329, 0.0027700
7: -0.0211434, -0.0125846, -0.0211373, -0.0125018, -0.0048238, 0.0047840
8: 0.9632125, 0.9877344, 0.9632300, 0.9879717, -0.0170035, 0.0170379
9: 0.0012580, 0.0084650, 0.0011883, 0.0084599, -0.0041969, 0.0042198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109555, upper bound: 0.0107728
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106063, upper bound: 0.0106248
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004487, 0.0009208, -0.0004491, 0.0009394, -0.0010402, 0.0010180
1: -0.0007803, 0.0027328, -0.0007824, 0.0027614, -0.0025935, 0.0025672
2: 0.0122473, 0.0175086, 0.0122045, 0.0175117, -0.0035802, 0.0036306
3: -0.0014174, 0.0025388, -0.0014496, 0.0025412, -0.0025787, 0.0026193
4: -0.0056871, -0.0020378, -0.0057168, -0.0020356, -0.0033336, 0.0033417
5: 0.0065233, 0.0104723, 0.0064912, 0.0104747, -0.0025635, 0.0026041
6: 0.0077338, 0.0104712, 0.0076905, 0.0104833, -0.0027495, 0.0027807
7: -0.0211338, -0.0125610, -0.0211389, -0.0124912, -0.0048737, 0.0047555
8: 0.9632401, 0.9878023, 0.9632255, 0.9880022, -0.0172089, 0.0169847
9: 0.0012380, 0.0084569, 0.0011793, 0.0084612, -0.0041729, 0.0042676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109555, upper bound: 0.0107427
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106003, upper bound: 0.0106003
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004561, 0.0008986, -0.0004877, 0.0012761, -0.0013905, 0.0010524
1: -0.0008151, 0.0026987, -0.0009628, 0.0032773, -0.0031669, 0.0028230
2: 0.0122983, 0.0175608, 0.0114318, 0.0177819, -0.0039406, 0.0044936
3: -0.0013791, 0.0025780, -0.0020306, 0.0027444, -0.0028487, 0.0032707
4: -0.0056517, -0.0020016, -0.0062527, -0.0018482, -0.0036824, 0.0039196
5: 0.0065616, 0.0105115, 0.0059112, 0.0106775, -0.0028339, 0.0032548
6: 0.0077854, 0.0104567, 0.0069080, 0.0107022, -0.0029167, 0.0035488
7: -0.0212188, -0.0126440, -0.0215792, -0.0112322, -0.0063324, 0.0053852
8: 0.9629965, 0.9875644, 0.9619640, 0.9916095, -0.0212319, 0.0186852
9: 0.0013080, 0.0085285, 0.0001191, 0.0088319, -0.0047179, 0.0054895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110732, upper bound: 0.0121477
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110666, upper bound: 0.0121491
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004514, 0.0008960, -0.0004878, 0.0012781, -0.0013948, 0.0010509
1: -0.0007929, 0.0026948, -0.0009633, 0.0032803, -0.0031895, 0.0028162
2: 0.0123043, 0.0175274, 0.0114274, 0.0177826, -0.0039253, 0.0045288
3: -0.0013746, 0.0025530, -0.0020340, 0.0027449, -0.0028368, 0.0032978
4: -0.0056476, -0.0020248, -0.0062558, -0.0018478, -0.0036793, 0.0039374
5: 0.0065661, 0.0104865, 0.0059079, 0.0106780, -0.0028213, 0.0032821
6: 0.0077915, 0.0104550, 0.0069035, 0.0107034, -0.0029119, 0.0035516
7: -0.0211644, -0.0126537, -0.0215802, -0.0112249, -0.0063782, 0.0053688
8: 0.9631524, 0.9875365, 0.9619610, 0.9916304, -0.0213982, 0.0186200
9: 0.0013162, 0.0084827, 0.0001130, 0.0088329, -0.0047026, 0.0055307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110751, upper bound: 0.0120945
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110689, upper bound: 0.0121066
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004561, 0.0008986, -0.0005000, 0.0012508, -0.0013690, 0.0010663
1: -0.0008151, 0.0026987, -0.0010204, 0.0032386, -0.0031359, 0.0028559
2: 0.0122983, 0.0175608, 0.0114898, 0.0178682, -0.0040169, 0.0044471
3: -0.0013791, 0.0025780, -0.0019870, 0.0028092, -0.0029110, 0.0032357
4: -0.0056517, -0.0020016, -0.0062125, -0.0017884, -0.0036820, 0.0038874
5: 0.0065616, 0.0105115, 0.0059547, 0.0107423, -0.0028957, 0.0032199
6: 0.0077854, 0.0104567, 0.0069667, 0.0106857, -0.0029003, 0.0034901
7: -0.0212188, -0.0126440, -0.0217197, -0.0113266, -0.0062567, 0.0055368
8: 0.9629965, 0.9875644, 0.9615612, 0.9913389, -0.0210149, 0.0190094
9: 0.0013080, 0.0085285, 0.0001986, 0.0089503, -0.0048490, 0.0054257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109329, upper bound: 0.0118567
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109025, upper bound: 0.0118548
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004514, 0.0008960, -0.0005001, 0.0012527, -0.0013729, 0.0010645
1: -0.0007929, 0.0026948, -0.0010209, 0.0032414, -0.0031588, 0.0028457
2: 0.0123043, 0.0175274, 0.0114856, 0.0178690, -0.0040012, 0.0044828
3: -0.0013746, 0.0025530, -0.0019902, 0.0028098, -0.0028988, 0.0032632
4: -0.0056476, -0.0020248, -0.0062154, -0.0017879, -0.0036772, 0.0039055
5: 0.0065661, 0.0104865, 0.0059516, 0.0107428, -0.0028833, 0.0032476
6: 0.0077915, 0.0104550, 0.0069625, 0.0106869, -0.0028954, 0.0034926
7: -0.0211644, -0.0126537, -0.0217210, -0.0113198, -0.0063033, 0.0055161
8: 0.9631524, 0.9875365, 0.9615577, 0.9913583, -0.0211836, 0.0189340
9: 0.0013162, 0.0084827, 0.0001929, 0.0089514, -0.0048251, 0.0054676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109367, upper bound: 0.0118269
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109129, upper bound: 0.0118145
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004675, 0.0008712, -0.0004877, 0.0012761, -0.0014070, 0.0010296
1: -0.0008685, 0.0026568, -0.0009628, 0.0032773, -0.0031977, 0.0028077
2: 0.0123611, 0.0176407, 0.0114318, 0.0177819, -0.0039176, 0.0045614
3: -0.0013319, 0.0026382, -0.0020306, 0.0027444, -0.0028315, 0.0033334
4: -0.0056082, -0.0019462, -0.0062527, -0.0018482, -0.0036665, 0.0039110
5: 0.0066087, 0.0105715, 0.0059112, 0.0106775, -0.0028167, 0.0033182
6: 0.0078490, 0.0104390, 0.0069080, 0.0107022, -0.0028532, 0.0035310
7: -0.0213490, -0.0127462, -0.0215792, -0.0112322, -0.0065290, 0.0053479
8: 0.9626234, 0.9872714, 0.9619640, 0.9916095, -0.0215229, 0.0185782
9: 0.0013941, 0.0086382, 0.0001191, 0.0088319, -0.0046864, 0.0056419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109339, upper bound: 0.0118007
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109004, upper bound: 0.0117786
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004633, 0.0008703, -0.0004878, 0.0012781, -0.0014127, 0.0010280
1: -0.0008485, 0.0026554, -0.0009633, 0.0032803, -0.0032256, 0.0027997
2: 0.0123632, 0.0176107, 0.0114274, 0.0177826, -0.0039007, 0.0046010
3: -0.0013303, 0.0026156, -0.0020340, 0.0027449, -0.0028183, 0.0033634
4: -0.0056067, -0.0019670, -0.0062558, -0.0018478, -0.0036623, 0.0039331
5: 0.0066103, 0.0105490, 0.0059079, 0.0106780, -0.0028029, 0.0033485
6: 0.0078512, 0.0104384, 0.0069035, 0.0107034, -0.0028523, 0.0035349
7: -0.0213001, -0.0127497, -0.0215802, -0.0112249, -0.0065774, 0.0053286
8: 0.9627634, 0.9872614, 0.9619610, 0.9916304, -0.0217138, 0.0185050
9: 0.0013970, 0.0085970, 0.0001130, 0.0088329, -0.0046688, 0.0056982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109388, upper bound: 0.0117681
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109137, upper bound: 0.0117581
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004675, 0.0008712, -0.0005000, 0.0012508, -0.0013780, 0.0010376
1: -0.0008685, 0.0026568, -0.0010204, 0.0032386, -0.0031321, 0.0028022
2: 0.0123611, 0.0176407, 0.0114898, 0.0178682, -0.0039170, 0.0044533
3: -0.0013319, 0.0026382, -0.0019870, 0.0028092, -0.0028416, 0.0032484
4: -0.0056082, -0.0019462, -0.0062125, -0.0017884, -0.0036458, 0.0038574
5: 0.0066087, 0.0105715, 0.0059547, 0.0107423, -0.0028271, 0.0032334
6: 0.0078490, 0.0104390, 0.0069667, 0.0106857, -0.0028367, 0.0034723
7: -0.0213490, -0.0127462, -0.0217197, -0.0113266, -0.0063573, 0.0054126
8: 0.9626234, 0.9872714, 0.9615612, 0.9913389, -0.0210289, 0.0185673
9: 0.0013941, 0.0086382, 0.0001986, 0.0089503, -0.0047361, 0.0054938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109242, upper bound: 0.0117902
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108860, upper bound: 0.0117644
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004633, 0.0008703, -0.0005001, 0.0012527, -0.0013826, 0.0010364
1: -0.0008485, 0.0026554, -0.0010209, 0.0032414, -0.0031583, 0.0027939
2: 0.0123632, 0.0176107, 0.0114856, 0.0178690, -0.0039023, 0.0044934
3: -0.0013303, 0.0026156, -0.0019902, 0.0028098, -0.0028272, 0.0032788
4: -0.0056067, -0.0019670, -0.0062154, -0.0017879, -0.0036429, 0.0038822
5: 0.0066103, 0.0105490, 0.0059516, 0.0107428, -0.0028128, 0.0032639
6: 0.0078512, 0.0104384, 0.0069625, 0.0106869, -0.0028358, 0.0034759
7: -0.0213001, -0.0127497, -0.0217210, -0.0113198, -0.0064056, 0.0053932
8: 0.9627634, 0.9872614, 0.9615577, 0.9913583, -0.0212172, 0.0184976
9: 0.0013970, 0.0085970, 0.0001929, 0.0089514, -0.0047191, 0.0055386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109308, upper bound: 0.0117586
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109017, upper bound: 0.0117412
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004561, 0.0008986, -0.0004740, 0.0013265, -0.0014461, 0.0010442
1: -0.0008151, 0.0026987, -0.0008988, 0.0033546, -0.0032792, 0.0028430
2: 0.0122983, 0.0175608, 0.0113161, 0.0176860, -0.0039606, 0.0046619
3: -0.0013791, 0.0025780, -0.0021177, 0.0026722, -0.0028524, 0.0033972
4: -0.0056517, -0.0020016, -0.0063330, -0.0019147, -0.0037346, 0.0040363
5: 0.0065616, 0.0105115, 0.0058243, 0.0106055, -0.0028356, 0.0033811
6: 0.0077854, 0.0104567, 0.0067907, 0.0107350, -0.0029495, 0.0036660
7: -0.0212188, -0.0126440, -0.0214229, -0.0110435, -0.0066065, 0.0052977
8: 0.9629965, 0.9875644, 0.9624118, 0.9921500, -0.0220173, 0.0187896
9: 0.0013080, 0.0085285, -0.0000397, 0.0087004, -0.0046594, 0.0057203

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107737, upper bound: 0.0114636
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106782, upper bound: 0.0114029
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004514, 0.0008960, -0.0004741, 0.0013296, -0.0014514, 0.0010424
1: -0.0007929, 0.0026948, -0.0008992, 0.0033593, -0.0033029, 0.0028364
2: 0.0123043, 0.0175274, 0.0113091, 0.0176867, -0.0039486, 0.0046986
3: -0.0013746, 0.0025530, -0.0021229, 0.0026728, -0.0028425, 0.0034255
4: -0.0056476, -0.0020248, -0.0063378, -0.0019143, -0.0037311, 0.0040551
5: 0.0065661, 0.0104865, 0.0058191, 0.0106060, -0.0028259, 0.0034096
6: 0.0077915, 0.0104550, 0.0067837, 0.0107369, -0.0029454, 0.0036713
7: -0.0211644, -0.0126537, -0.0214240, -0.0110322, -0.0066549, 0.0052794
8: 0.9631524, 0.9875365, 0.9624085, 0.9921825, -0.0221910, 0.0187306
9: 0.0013162, 0.0084827, -0.0000493, 0.0087013, -0.0046332, 0.0057637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107737, upper bound: 0.0114329
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106794, upper bound: 0.0113747
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004561, 0.0008986, -0.0004869, 0.0013027, -0.0014256, 0.0010574
1: -0.0008151, 0.0026987, -0.0009590, 0.0033180, -0.0032445, 0.0028714
2: 0.0122983, 0.0175608, 0.0113709, 0.0177762, -0.0040165, 0.0046098
3: -0.0013791, 0.0025780, -0.0020764, 0.0027400, -0.0029000, 0.0033581
4: -0.0056517, -0.0020016, -0.0062950, -0.0018522, -0.0037325, 0.0040002
5: 0.0065616, 0.0105115, 0.0058655, 0.0106732, -0.0028841, 0.0033420
6: 0.0077854, 0.0104567, 0.0068463, 0.0107194, -0.0029340, 0.0036105
7: -0.0212188, -0.0126440, -0.0215698, -0.0111329, -0.0065218, 0.0054517
8: 0.9629965, 0.9875644, 0.9619908, 0.9918939, -0.0217745, 0.0190344
9: 0.0013080, 0.0085285, 0.0000355, 0.0088241, -0.0047964, 0.0056489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106190, upper bound: 0.0112029
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103783, upper bound: 0.0110541
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004514, 0.0008960, -0.0004870, 0.0013061, -0.0014307, 0.0010551
1: -0.0007929, 0.0026948, -0.0009594, 0.0033232, -0.0032677, 0.0028633
2: 0.0123043, 0.0175274, 0.0113631, 0.0177768, -0.0040017, 0.0046459
3: -0.0013746, 0.0025530, -0.0020823, 0.0027405, -0.0028881, 0.0033859
4: -0.0056476, -0.0020248, -0.0063004, -0.0018518, -0.0037290, 0.0040186
5: 0.0065661, 0.0104865, 0.0058596, 0.0106737, -0.0028718, 0.0033700
6: 0.0077915, 0.0104550, 0.0068383, 0.0107216, -0.0029302, 0.0036167
7: -0.0211644, -0.0126537, -0.0215708, -0.0111201, -0.0065690, 0.0054240
8: 0.9631524, 0.9875365, 0.9619880, 0.9919305, -0.0219451, 0.0189659
9: 0.0013162, 0.0084827, 0.0000248, 0.0088249, -0.0047654, 0.0056914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106191, upper bound: 0.0111567
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103790, upper bound: 0.0110213
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004675, 0.0008712, -0.0004740, 0.0013265, -0.0014625, 0.0010214
1: -0.0008685, 0.0026568, -0.0008988, 0.0033546, -0.0033100, 0.0028277
2: 0.0123611, 0.0176407, 0.0113161, 0.0176860, -0.0039377, 0.0047297
3: -0.0013319, 0.0026382, -0.0021177, 0.0026722, -0.0028351, 0.0034599
4: -0.0056082, -0.0019462, -0.0063330, -0.0019147, -0.0036934, 0.0040277
5: 0.0066087, 0.0105715, 0.0058243, 0.0106055, -0.0028184, 0.0034444
6: 0.0078490, 0.0104390, 0.0067907, 0.0107350, -0.0028860, 0.0036482
7: -0.0213490, -0.0127462, -0.0214229, -0.0110435, -0.0068031, 0.0052604
8: 0.9626234, 0.9872714, 0.9624118, 0.9921500, -0.0223083, 0.0186826
9: 0.0013941, 0.0086382, -0.0000397, 0.0087004, -0.0046279, 0.0058728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106075, upper bound: 0.0111537
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103975, upper bound: 0.0109709
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004633, 0.0008703, -0.0004741, 0.0013296, -0.0014693, 0.0010195
1: -0.0008485, 0.0026554, -0.0008992, 0.0033593, -0.0033390, 0.0028199
2: 0.0123632, 0.0176107, 0.0113091, 0.0176867, -0.0039240, 0.0047708
3: -0.0013303, 0.0026156, -0.0021229, 0.0026728, -0.0028240, 0.0034910
4: -0.0056067, -0.0019670, -0.0063378, -0.0019143, -0.0036924, 0.0040509
5: 0.0066103, 0.0105490, 0.0058191, 0.0106060, -0.0028074, 0.0034759
6: 0.0078512, 0.0104384, 0.0067837, 0.0107369, -0.0028858, 0.0036547
7: -0.0213001, -0.0127497, -0.0214240, -0.0110322, -0.0068541, 0.0052393
8: 0.9627634, 0.9872614, 0.9624085, 0.9921825, -0.0225066, 0.0186156
9: 0.0013970, 0.0085970, -0.0000493, 0.0087013, -0.0045994, 0.0059312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106075, upper bound: 0.0111339
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103990, upper bound: 0.0109607
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004675, 0.0008712, -0.0004869, 0.0013027, -0.0014346, 0.0010292
1: -0.0008685, 0.0026568, -0.0009590, 0.0033180, -0.0032449, 0.0028266
2: 0.0123611, 0.0176407, 0.0113709, 0.0177762, -0.0039456, 0.0046222
3: -0.0013319, 0.0026382, -0.0020764, 0.0027400, -0.0028471, 0.0033754
4: -0.0056082, -0.0019462, -0.0062950, -0.0018522, -0.0036989, 0.0039746
5: 0.0066087, 0.0105715, 0.0058655, 0.0106732, -0.0028310, 0.0033603
6: 0.0078490, 0.0104390, 0.0068463, 0.0107194, -0.0028704, 0.0035927
7: -0.0213490, -0.0127462, -0.0215698, -0.0111329, -0.0066327, 0.0053264
8: 0.9626234, 0.9872714, 0.9619908, 0.9918939, -0.0218177, 0.0187054
9: 0.0013941, 0.0086382, 0.0000355, 0.0088241, -0.0046783, 0.0057256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105949, upper bound: 0.0111257
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103310, upper bound: 0.0109487
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004633, 0.0008703, -0.0004870, 0.0013061, -0.0014404, 0.0010279
1: -0.0008485, 0.0026554, -0.0009594, 0.0033232, -0.0032716, 0.0028175
2: 0.0123632, 0.0176107, 0.0113631, 0.0177768, -0.0039314, 0.0046631
3: -0.0013303, 0.0026156, -0.0020823, 0.0027405, -0.0028366, 0.0034064
4: -0.0056067, -0.0019670, -0.0063004, -0.0018518, -0.0036960, 0.0039999
5: 0.0066103, 0.0105490, 0.0058596, 0.0106737, -0.0028204, 0.0033912
6: 0.0078512, 0.0104384, 0.0068383, 0.0107216, -0.0028705, 0.0036000
7: -0.0213001, -0.0127497, -0.0215708, -0.0111201, -0.0066820, 0.0053064
8: 0.9627634, 0.9872614, 0.9619880, 0.9919305, -0.0220094, 0.0186445
9: 0.0013970, 0.0085970, 0.0000248, 0.0088249, -0.0046514, 0.0057714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105954, upper bound: 0.0111035
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103334, upper bound: 0.0109354
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004304, 0.0009698, -0.0004879, 0.0012858, -0.0013893, 0.0011321
1: -0.0006945, 0.0028079, -0.0009639, 0.0032921, -0.0032340, 0.0030022
2: 0.0121348, 0.0173801, 0.0114097, 0.0177836, -0.0042054, 0.0045813
3: -0.0015020, 0.0024422, -0.0020473, 0.0027456, -0.0030458, 0.0033300
4: -0.0057651, -0.0021269, -0.0062681, -0.0018471, -0.0038774, 0.0040299
5: 0.0064389, 0.0103759, 0.0058946, 0.0106788, -0.0030305, 0.0033132
6: 0.0076199, 0.0105030, 0.0068855, 0.0107084, -0.0030886, 0.0036175
7: -0.0209244, -0.0123776, -0.0215819, -0.0111960, -0.0062816, 0.0058020
8: 0.9638399, 0.9883276, 0.9619563, 0.9917130, -0.0216542, 0.0199265
9: 0.0010836, 0.0082806, 0.0000887, 0.0088343, -0.0050703, 0.0055094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108449, upper bound: 0.0116839
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108449, upper bound: 0.0116839
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004199, 0.0010543, -0.0004828, 0.0012832, -0.0013747, 0.0012097
1: -0.0006456, 0.0029374, -0.0009401, 0.0032882, -0.0033657, 0.0030912
2: 0.0119409, 0.0173068, 0.0114156, 0.0177479, -0.0043471, 0.0047298
3: -0.0016479, 0.0023871, -0.0020429, 0.0027188, -0.0031520, 0.0034209
4: -0.0058996, -0.0021778, -0.0062640, -0.0018718, -0.0039630, 0.0040862
5: 0.0062933, 0.0103209, 0.0058990, 0.0106520, -0.0031363, 0.0034020
6: 0.0074235, 0.0105580, 0.0068915, 0.0107068, -0.0032833, 0.0036665
7: -0.0208050, -0.0120616, -0.0215237, -0.0112056, -0.0061613, 0.0060502
8: 0.9641820, 0.9892331, 0.9621230, 0.9916855, -0.0224044, 0.0205695
9: 0.0008176, 0.0081801, 0.0000968, 0.0087853, -0.0052775, 0.0055051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106363, upper bound: 0.0113888
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105593, upper bound: 0.0113888
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004304, 0.0009698, -0.0004742, 0.0013369, -0.0014307, 0.0011128
1: -0.0006945, 0.0028079, -0.0008998, 0.0033706, -0.0032632, 0.0029386
2: 0.0121348, 0.0173801, 0.0112922, 0.0176875, -0.0040951, 0.0046133
3: -0.0015020, 0.0024422, -0.0021356, 0.0026734, -0.0029507, 0.0033494
4: -0.0057651, -0.0021269, -0.0063495, -0.0019137, -0.0038514, 0.0040868
5: 0.0064389, 0.0103759, 0.0058064, 0.0106067, -0.0029340, 0.0033323
6: 0.0076199, 0.0105030, 0.0067666, 0.0107417, -0.0031218, 0.0037365
7: -0.0209244, -0.0123776, -0.0214253, -0.0110046, -0.0063318, 0.0055247
8: 0.9638399, 0.9883276, 0.9624048, 0.9922614, -0.0218161, 0.0194265
9: 0.0010836, 0.0082806, -0.0000725, 0.0087024, -0.0048445, 0.0055184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108288, upper bound: 0.0116354
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108288, upper bound: 0.0116354
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004199, 0.0010543, -0.0004692, 0.0013344, -0.0014149, 0.0011903
1: -0.0006456, 0.0029374, -0.0008761, 0.0033667, -0.0034008, 0.0030306
2: 0.0119409, 0.0173068, 0.0112981, 0.0176521, -0.0042387, 0.0047722
3: -0.0016479, 0.0023871, -0.0021312, 0.0026467, -0.0030631, 0.0034450
4: -0.0058996, -0.0021778, -0.0063455, -0.0019383, -0.0039516, 0.0041677
5: 0.0062933, 0.0103209, 0.0058108, 0.0105801, -0.0030460, 0.0034256
6: 0.0074235, 0.0105580, 0.0067725, 0.0107400, -0.0033166, 0.0037855
7: -0.0208050, -0.0120616, -0.0213676, -0.0110142, -0.0061855, 0.0057802
8: 0.9641820, 0.9892331, 0.9625703, 0.9922340, -0.0226112, 0.0200812
9: 0.0008176, 0.0081801, -0.0000645, 0.0086538, -0.0050597, 0.0055067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106133, upper bound: 0.0112967
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104797, upper bound: 0.0112886
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004495, 0.0009145, -0.0004877, 0.0012761, -0.0014019, 0.0010811
1: -0.0007843, 0.0027231, -0.0009628, 0.0032773, -0.0032379, 0.0029320
2: 0.0122619, 0.0175145, 0.0114318, 0.0177819, -0.0041038, 0.0046028
3: -0.0014065, 0.0025433, -0.0020306, 0.0027444, -0.0029715, 0.0033535
4: -0.0056770, -0.0020337, -0.0062527, -0.0018482, -0.0037957, 0.0039729
5: 0.0065342, 0.0104768, 0.0059112, 0.0106775, -0.0029564, 0.0033375
6: 0.0077485, 0.0104671, 0.0069080, 0.0107022, -0.0029536, 0.0035591
7: -0.0211434, -0.0125846, -0.0215792, -0.0112322, -0.0064904, 0.0056512
8: 0.9632125, 0.9877344, 0.9619640, 0.9916095, -0.0217383, 0.0194473
9: 0.0012580, 0.0084650, 0.0001191, 0.0088319, -0.0049419, 0.0056535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104720, upper bound: 0.0111336
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102741, upper bound: 0.0110486
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004487, 0.0009208, -0.0004878, 0.0012781, -0.0014081, 0.0010861
1: -0.0007803, 0.0027328, -0.0009633, 0.0032803, -0.0032705, 0.0029265
2: 0.0122473, 0.0175086, 0.0114274, 0.0177826, -0.0040906, 0.0046500
3: -0.0014174, 0.0025388, -0.0020340, 0.0027449, -0.0029611, 0.0033901
4: -0.0056871, -0.0020378, -0.0062558, -0.0018478, -0.0037940, 0.0040034
5: 0.0065233, 0.0104723, 0.0059079, 0.0106780, -0.0029454, 0.0033743
6: 0.0077338, 0.0104712, 0.0069035, 0.0107034, -0.0029696, 0.0035677
7: -0.0211338, -0.0125610, -0.0215802, -0.0112249, -0.0065533, 0.0056381
8: 0.9632401, 0.9878023, 0.9619610, 0.9916304, -0.0219580, 0.0193917
9: 0.0012380, 0.0084569, 0.0001130, 0.0088329, -0.0049294, 0.0057074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104720, upper bound: 0.0110829
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102741, upper bound: 0.0109946
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004495, 0.0009145, -0.0004740, 0.0013265, -0.0014439, 0.0010623
1: -0.0007843, 0.0027231, -0.0008988, 0.0033546, -0.0032692, 0.0028710
2: 0.0122619, 0.0175145, 0.0113161, 0.0176860, -0.0039975, 0.0046444
3: -0.0014065, 0.0025433, -0.0021177, 0.0026722, -0.0028789, 0.0033872
4: -0.0056770, -0.0020337, -0.0063330, -0.0019147, -0.0037622, 0.0040245
5: 0.0065342, 0.0104768, 0.0058243, 0.0106055, -0.0028625, 0.0033711
6: 0.0077485, 0.0104671, 0.0067907, 0.0107350, -0.0029864, 0.0036763
7: -0.0211434, -0.0125846, -0.0214229, -0.0110435, -0.0065493, 0.0053819
8: 0.9632125, 0.9877344, 0.9624118, 0.9921500, -0.0219341, 0.0189686
9: 0.0012580, 0.0084650, -0.0000397, 0.0087004, -0.0047222, 0.0056807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104576, upper bound: 0.0110396
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101993, upper bound: 0.0108584
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004487, 0.0009208, -0.0004741, 0.0013296, -0.0014502, 0.0010663
1: -0.0007803, 0.0027328, -0.0008992, 0.0033593, -0.0033014, 0.0028698
2: 0.0122473, 0.0175086, 0.0113091, 0.0176867, -0.0039880, 0.0046950
3: -0.0014174, 0.0025388, -0.0021229, 0.0026728, -0.0028713, 0.0034221
4: -0.0056871, -0.0020378, -0.0063378, -0.0019143, -0.0037728, 0.0040564
5: 0.0065233, 0.0104723, 0.0058191, 0.0106060, -0.0028548, 0.0034058
6: 0.0077338, 0.0104712, 0.0067837, 0.0107369, -0.0030031, 0.0036875
7: -0.0211338, -0.0125610, -0.0214240, -0.0110322, -0.0066072, 0.0053674
8: 0.9632401, 0.9878023, 0.9624085, 0.9921825, -0.0221672, 0.0189271
9: 0.0012380, 0.0084569, -0.0000493, 0.0087013, -0.0047109, 0.0057493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104576, upper bound: 0.0110145
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101993, upper bound: 0.0108496
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004304, 0.0009698, -0.0005002, 0.0012606, -0.0013676, 0.0011460
1: -0.0006945, 0.0028079, -0.0010216, 0.0032535, -0.0032030, 0.0030342
2: 0.0121348, 0.0173801, 0.0114675, 0.0178700, -0.0042812, 0.0045349
3: -0.0015020, 0.0024422, -0.0020039, 0.0028106, -0.0031085, 0.0032950
4: -0.0057651, -0.0021269, -0.0062280, -0.0017872, -0.0038767, 0.0039976
5: 0.0064389, 0.0103759, 0.0059380, 0.0107436, -0.0030929, 0.0032784
6: 0.0076199, 0.0105030, 0.0069440, 0.0106921, -0.0030722, 0.0035590
7: -0.0209244, -0.0123776, -0.0217226, -0.0112902, -0.0062059, 0.0059533
8: 0.9638399, 0.9883276, 0.9615530, 0.9914432, -0.0214374, 0.0202471
9: 0.0010836, 0.0082806, 0.0001680, 0.0089527, -0.0052019, 0.0054457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104934, upper bound: 0.0112963
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104934, upper bound: 0.0112963
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004199, 0.0010543, -0.0004955, 0.0012580, -0.0013527, 0.0012231
1: -0.0006456, 0.0029374, -0.0009992, 0.0032496, -0.0033364, 0.0031256
2: 0.0119409, 0.0173068, 0.0114733, 0.0178364, -0.0044238, 0.0046858
3: -0.0016479, 0.0023871, -0.0019994, 0.0027853, -0.0032172, 0.0033878
4: -0.0058996, -0.0021778, -0.0062239, -0.0018105, -0.0039627, 0.0040462
5: 0.0062933, 0.0103209, 0.0059424, 0.0107184, -0.0032012, 0.0033690
6: 0.0074235, 0.0105580, 0.0069500, 0.0106904, -0.0032669, 0.0036080
7: -0.0208050, -0.0120616, -0.0216679, -0.0112998, -0.0060897, 0.0062042
8: 0.9641820, 0.9892331, 0.9617098, 0.9914159, -0.0221991, 0.0209139
9: 0.0008176, 0.0081801, 0.0001760, 0.0089067, -0.0054090, 0.0054448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102965, upper bound: 0.0110550
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102806, upper bound: 0.0110619
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004304, 0.0009698, -0.0004871, 0.0013133, -0.0014099, 0.0011262
1: -0.0006945, 0.0028079, -0.0009600, 0.0033343, -0.0032313, 0.0029692
2: 0.0121348, 0.0173801, 0.0113465, 0.0177776, -0.0041668, 0.0045656
3: -0.0015020, 0.0024422, -0.0020948, 0.0027411, -0.0030153, 0.0033136
4: -0.0057651, -0.0021269, -0.0063119, -0.0018512, -0.0038617, 0.0040537
5: 0.0064389, 0.0103759, 0.0058471, 0.0106743, -0.0029988, 0.0032965
6: 0.0076199, 0.0105030, 0.0068215, 0.0107263, -0.0031065, 0.0036815
7: -0.0209244, -0.0123776, -0.0215721, -0.0110930, -0.0062542, 0.0056752
8: 0.9638399, 0.9883276, 0.9619842, 0.9920081, -0.0215935, 0.0197363
9: 0.0010836, 0.0082806, 0.0000019, 0.0088260, -0.0049776, 0.0054529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103882, upper bound: 0.0111206
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103882, upper bound: 0.0111206
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004199, 0.0010543, -0.0004822, 0.0013108, -0.0013935, 0.0012034
1: -0.0006456, 0.0029374, -0.0009369, 0.0033305, -0.0033700, 0.0030632
2: 0.0119409, 0.0173068, 0.0113523, 0.0177431, -0.0043143, 0.0047262
3: -0.0016479, 0.0023871, -0.0020905, 0.0027152, -0.0031266, 0.0034104
4: -0.0058996, -0.0021778, -0.0063079, -0.0018752, -0.0039500, 0.0041301
5: 0.0062933, 0.0103209, 0.0058515, 0.0106484, -0.0031102, 0.0033911
6: 0.0074235, 0.0105580, 0.0068274, 0.0107247, -0.0033012, 0.0037306
7: -0.0208050, -0.0120616, -0.0215159, -0.0111025, -0.0061104, 0.0059328
8: 0.9641820, 0.9892331, 0.9621453, 0.9919809, -0.0223961, 0.0204176
9: 0.0008176, 0.0081801, 0.0000099, 0.0087787, -0.0051917, 0.0054435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102130, upper bound: 0.0109143
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101719, upper bound: 0.0109143
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004495, 0.0009145, -0.0005000, 0.0012508, -0.0013730, 0.0010881
1: -0.0007843, 0.0027231, -0.0010204, 0.0032386, -0.0031809, 0.0029316
2: 0.0122619, 0.0175145, 0.0114898, 0.0178682, -0.0041109, 0.0045201
3: -0.0014065, 0.0025433, -0.0019870, 0.0028092, -0.0029873, 0.0032949
4: -0.0056770, -0.0020337, -0.0062125, -0.0017884, -0.0037803, 0.0039271
5: 0.0065342, 0.0104768, 0.0059547, 0.0107423, -0.0029726, 0.0032795
6: 0.0077485, 0.0104671, 0.0069667, 0.0106857, -0.0029372, 0.0035004
7: -0.0211434, -0.0125846, -0.0217197, -0.0113266, -0.0063249, 0.0057285
8: 0.9632125, 0.9877344, 0.9615612, 0.9913389, -0.0213494, 0.0194723
9: 0.0012580, 0.0084650, 0.0001986, 0.0089503, -0.0050021, 0.0055090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104594, upper bound: 0.0111256
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102364, upper bound: 0.0110264
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004487, 0.0009208, -0.0005001, 0.0012527, -0.0013784, 0.0010931
1: -0.0007803, 0.0027328, -0.0010209, 0.0032414, -0.0032107, 0.0029247
2: 0.0122473, 0.0175086, 0.0114856, 0.0178690, -0.0040982, 0.0045667
3: -0.0014174, 0.0025388, -0.0019902, 0.0028098, -0.0029745, 0.0033301
4: -0.0056871, -0.0020378, -0.0062154, -0.0017879, -0.0037787, 0.0039591
5: 0.0065233, 0.0104723, 0.0059516, 0.0107428, -0.0029598, 0.0033145
6: 0.0077338, 0.0104712, 0.0069625, 0.0106869, -0.0029531, 0.0035087
7: -0.0211338, -0.0125610, -0.0217210, -0.0113198, -0.0063816, 0.0057124
8: 0.9632401, 0.9878023, 0.9615577, 0.9913583, -0.0215665, 0.0194120
9: 0.0012380, 0.0084569, 0.0001929, 0.0089514, -0.0049879, 0.0055596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104593, upper bound: 0.0110759
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102311, upper bound: 0.0109598
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004495, 0.0009145, -0.0004869, 0.0013027, -0.0014153, 0.0010687
1: -0.0007843, 0.0027231, -0.0009590, 0.0033180, -0.0032055, 0.0028701
2: 0.0122619, 0.0175145, 0.0113709, 0.0177762, -0.0040010, 0.0045432
3: -0.0014065, 0.0025433, -0.0020764, 0.0027400, -0.0028879, 0.0033067
4: -0.0056770, -0.0020337, -0.0062950, -0.0018522, -0.0037626, 0.0039756
5: 0.0065342, 0.0104768, 0.0058655, 0.0106732, -0.0028730, 0.0032903
6: 0.0077485, 0.0104671, 0.0068463, 0.0107194, -0.0029709, 0.0036208
7: -0.0211434, -0.0125846, -0.0215698, -0.0111329, -0.0063839, 0.0054510
8: 0.9632125, 0.9877344, 0.9619908, 0.9918939, -0.0214735, 0.0189772
9: 0.0012580, 0.0084650, 0.0000355, 0.0088241, -0.0047756, 0.0055336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104439, upper bound: 0.0110258
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101228, upper bound: 0.0108117
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004487, 0.0009208, -0.0004870, 0.0013061, -0.0014217, 0.0010730
1: -0.0007803, 0.0027328, -0.0009594, 0.0033232, -0.0032352, 0.0028666
2: 0.0122473, 0.0175086, 0.0113631, 0.0177768, -0.0039897, 0.0045917
3: -0.0014174, 0.0025388, -0.0020823, 0.0027405, -0.0028783, 0.0033420
4: -0.0056871, -0.0020378, -0.0063004, -0.0018518, -0.0037610, 0.0040083
5: 0.0065233, 0.0104723, 0.0058596, 0.0106737, -0.0028628, 0.0033254
6: 0.0077338, 0.0104712, 0.0068383, 0.0107216, -0.0029878, 0.0036328
7: -0.0211338, -0.0125610, -0.0215708, -0.0111201, -0.0064396, 0.0054334
8: 0.9632401, 0.9878023, 0.9619880, 0.9919305, -0.0216955, 0.0189292
9: 0.0012380, 0.0084569, 0.0000248, 0.0088249, -0.0047624, 0.0055862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104439, upper bound: 0.0109933
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101195, upper bound: 0.0107904
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0012836, -0.0004521, 0.0009263, -0.0010725, 0.0014003
1: -0.0009312, 0.0032888, -0.0007963, 0.0027411, -0.0028353, 0.0032244
2: 0.0114147, 0.0177346, 0.0122348, 0.0175325, -0.0045737, 0.0039532
3: -0.0020435, 0.0027087, -0.0014268, 0.0025568, -0.0033261, 0.0028564
4: -0.0062646, -0.0018811, -0.0056957, -0.0020212, -0.0039974, 0.0037071
5: 0.0058984, 0.0106420, 0.0065139, 0.0104903, -0.0033098, 0.0028412
6: 0.0068906, 0.0107070, 0.0077211, 0.0104747, -0.0035841, 0.0029859
7: -0.0215020, -0.0112042, -0.0211728, -0.0125405, -0.0053979, 0.0063920
8: 0.9621851, 0.9916895, 0.9631283, 0.9878607, -0.0187539, 0.0216137
9: 0.0000956, 0.0087670, 0.0012209, 0.0084898, -0.0055518, 0.0047246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121477, upper bound: 0.0110732
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120945, upper bound: 0.0110751
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004705, 0.0013701, -0.0004469, 0.0009237, -0.0010798, 0.0014845
1: -0.0008822, 0.0034213, -0.0007719, 0.0027372, -0.0029549, 0.0033239
2: 0.0112162, 0.0176612, 0.0122407, 0.0174960, -0.0047262, 0.0041036
3: -0.0021928, 0.0026536, -0.0014224, 0.0025294, -0.0034457, 0.0029548
4: -0.0064022, -0.0019320, -0.0056917, -0.0020465, -0.0040946, 0.0037597
5: 0.0057494, 0.0105869, 0.0065184, 0.0104629, -0.0034296, 0.0029374
6: 0.0066896, 0.0107632, 0.0077271, 0.0104731, -0.0037834, 0.0030361
7: -0.0213824, -0.0108808, -0.0211133, -0.0125501, -0.0054787, 0.0066806
8: 0.9625279, 0.9926161, 0.9632988, 0.9878333, -0.0194699, 0.0223229
9: -0.0001767, 0.0086663, 0.0012289, 0.0084397, -0.0057938, 0.0048140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121491, upper bound: 0.0110666
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121066, upper bound: 0.0110689
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0012836, -0.0004639, 0.0009006, -0.0010496, 0.0014182
1: -0.0009312, 0.0032888, -0.0008517, 0.0027019, -0.0028188, 0.0032602
2: 0.0114147, 0.0177346, 0.0122936, 0.0176155, -0.0046437, 0.0039286
3: -0.0020435, 0.0027087, -0.0013826, 0.0026192, -0.0033892, 0.0028379
4: -0.0062646, -0.0018811, -0.0056549, -0.0019637, -0.0039904, 0.0036901
5: 0.0058984, 0.0106420, 0.0065581, 0.0105526, -0.0033739, 0.0028227
6: 0.0068906, 0.0107070, 0.0077807, 0.0104581, -0.0035674, 0.0029263
7: -0.0215020, -0.0112042, -0.0213079, -0.0126364, -0.0053578, 0.0065902
8: 0.9621851, 0.9916895, 0.9627411, 0.9875861, -0.0186392, 0.0219189
9: 0.0000956, 0.0087670, 0.0013016, 0.0086036, -0.0057194, 0.0046909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118007, upper bound: 0.0109339
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117681, upper bound: 0.0109388
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004705, 0.0013701, -0.0004590, 0.0008980, -0.0010568, 0.0015022
1: -0.0008822, 0.0034213, -0.0008284, 0.0026979, -0.0029395, 0.0033597
2: 0.0112162, 0.0176612, 0.0122996, 0.0175806, -0.0048044, 0.0040805
3: -0.0021928, 0.0026536, -0.0013781, 0.0025929, -0.0035138, 0.0029375
4: -0.0064022, -0.0019320, -0.0056508, -0.0019879, -0.0040895, 0.0037188
5: 0.0057494, 0.0105869, 0.0065625, 0.0105264, -0.0034984, 0.0029201
6: 0.0066896, 0.0107632, 0.0077867, 0.0104564, -0.0037667, 0.0029765
7: -0.0213824, -0.0108808, -0.0212510, -0.0126461, -0.0054411, 0.0068725
8: 0.9625279, 0.9926161, 0.9629042, 0.9875584, -0.0193622, 0.0226601
9: -0.0001767, 0.0086663, 0.0013097, 0.0085556, -0.0059543, 0.0047824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117786, upper bound: 0.0109004
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117581, upper bound: 0.0109137
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004932, 0.0012584, -0.0004521, 0.0009263, -0.0010867, 0.0013787
1: -0.0009886, 0.0032502, -0.0007963, 0.0027411, -0.0028641, 0.0031932
2: 0.0114724, 0.0178205, 0.0122348, 0.0175325, -0.0045270, 0.0040257
3: -0.0020001, 0.0027734, -0.0014268, 0.0025568, -0.0032910, 0.0029175
4: -0.0062245, -0.0018215, -0.0056957, -0.0020212, -0.0039650, 0.0037040
5: 0.0059417, 0.0107065, 0.0065139, 0.0104903, -0.0032747, 0.0029021
6: 0.0069491, 0.0106907, 0.0077211, 0.0104747, -0.0035256, 0.0029695
7: -0.0216420, -0.0112983, -0.0211728, -0.0125405, -0.0055495, 0.0063157
8: 0.9617841, 0.9914199, 0.9631283, 0.9878607, -0.0190514, 0.0213953
9: 0.0001748, 0.0088849, 0.0012209, 0.0084898, -0.0054876, 0.0048556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118567, upper bound: 0.0109329
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118269, upper bound: 0.0109367
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004835, 0.0013443, -0.0004469, 0.0009237, -0.0010940, 0.0014610
1: -0.0009430, 0.0033818, -0.0007719, 0.0027372, -0.0029787, 0.0032867
2: 0.0112754, 0.0177522, 0.0122407, 0.0174960, -0.0046704, 0.0041567
3: -0.0021483, 0.0027220, -0.0014224, 0.0025294, -0.0034038, 0.0030034
4: -0.0063612, -0.0018688, -0.0056917, -0.0020465, -0.0040559, 0.0038228
5: 0.0057938, 0.0106552, 0.0065184, 0.0104629, -0.0033877, 0.0029872
6: 0.0067496, 0.0107465, 0.0077271, 0.0104731, -0.0037235, 0.0030194
7: -0.0215307, -0.0109773, -0.0211133, -0.0125501, -0.0056365, 0.0065896
8: 0.9621028, 0.9923397, 0.9632988, 0.9878333, -0.0197040, 0.0220624
9: -0.0000955, 0.0087912, 0.0012289, 0.0084397, -0.0057172, 0.0049490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118548, upper bound: 0.0109025
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118145, upper bound: 0.0109129
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004932, 0.0012584, -0.0004639, 0.0009006, -0.0010582, 0.0013881
1: -0.0009886, 0.0032502, -0.0008517, 0.0027019, -0.0028131, 0.0031926
2: 0.0114724, 0.0178205, 0.0122936, 0.0176155, -0.0045368, 0.0039301
3: -0.0020001, 0.0027734, -0.0013826, 0.0026192, -0.0033058, 0.0028469
4: -0.0062245, -0.0018215, -0.0056549, -0.0019637, -0.0039409, 0.0036668
5: 0.0059417, 0.0107065, 0.0065581, 0.0105526, -0.0032902, 0.0028324
6: 0.0069491, 0.0106907, 0.0077807, 0.0104581, -0.0035090, 0.0029099
7: -0.0216420, -0.0112983, -0.0213079, -0.0126364, -0.0054218, 0.0064177
8: 0.9617841, 0.9914199, 0.9627411, 0.9875861, -0.0186301, 0.0214292
9: 0.0001748, 0.0088849, 0.0013016, 0.0086036, -0.0055581, 0.0047396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117902, upper bound: 0.0109271
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117586, upper bound: 0.0109311
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004835, 0.0013443, -0.0004590, 0.0008980, -0.0010648, 0.0014712
1: -0.0009430, 0.0033818, -0.0008284, 0.0026979, -0.0029277, 0.0032892
2: 0.0112754, 0.0177522, 0.0122996, 0.0175806, -0.0046864, 0.0040798
3: -0.0021483, 0.0027220, -0.0013781, 0.0025929, -0.0034225, 0.0029418
4: -0.0063612, -0.0018688, -0.0056508, -0.0019879, -0.0040360, 0.0037820
5: 0.0057938, 0.0106552, 0.0065625, 0.0105264, -0.0034070, 0.0029247
6: 0.0067496, 0.0107465, 0.0077867, 0.0104564, -0.0037068, 0.0029597
7: -0.0215307, -0.0109773, -0.0212510, -0.0126461, -0.0054979, 0.0066991
8: 0.9621028, 0.9923397, 0.9629042, 0.9875584, -0.0193442, 0.0221244
9: -0.0000955, 0.0087912, 0.0013097, 0.0085556, -0.0057951, 0.0048223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117644, upper bound: 0.0108875
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117412, upper bound: 0.0109023
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0012836, -0.0004370, 0.0009720, -0.0011245, 0.0013964
1: -0.0009312, 0.0032888, -0.0007255, 0.0028113, -0.0029646, 0.0032683
2: 0.0114147, 0.0177346, 0.0121298, 0.0174265, -0.0046324, 0.0041468
3: -0.0020435, 0.0027087, -0.0015058, 0.0024771, -0.0033695, 0.0030020
4: -0.0062646, -0.0018811, -0.0057686, -0.0020947, -0.0040650, 0.0038414
5: 0.0058984, 0.0106420, 0.0064351, 0.0104108, -0.0033531, 0.0029865
6: 0.0068906, 0.0107070, 0.0076148, 0.0105045, -0.0036139, 0.0030922
7: -0.0215020, -0.0112042, -0.0210001, -0.0123694, -0.0057134, 0.0063694
8: 0.9621851, 0.9916895, 0.9636232, 0.9883512, -0.0196580, 0.0218881
9: 0.0000956, 0.0087670, 0.0010767, 0.0083443, -0.0055798, 0.0049903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116839, upper bound: 0.0108449
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116839, upper bound: 0.0108449
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004705, 0.0013701, -0.0004318, 0.0009695, -0.0011318, 0.0014805
1: -0.0008822, 0.0034213, -0.0007014, 0.0028074, -0.0030852, 0.0033664
2: 0.0112162, 0.0176612, 0.0121356, 0.0173904, -0.0047821, 0.0042986
3: -0.0021928, 0.0026536, -0.0015014, 0.0024499, -0.0034859, 0.0031015
4: -0.0064022, -0.0019320, -0.0057646, -0.0021198, -0.0041614, 0.0038326
5: 0.0057494, 0.0105869, 0.0064395, 0.0103837, -0.0034694, 0.0030838
6: 0.0066896, 0.0107632, 0.0076207, 0.0105028, -0.0038132, 0.0031426
7: -0.0213824, -0.0108808, -0.0209412, -0.0123789, -0.0057965, 0.0066469
8: 0.9625279, 0.9926161, 0.9637918, 0.9883239, -0.0203804, 0.0225845
9: -0.0001767, 0.0086663, 0.0010847, 0.0082947, -0.0058068, 0.0050816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114818, upper bound: 0.0105593
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113888, upper bound: 0.0105593
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0012836, -0.0004493, 0.0009464, -0.0011050, 0.0014136
1: -0.0009312, 0.0032888, -0.0007832, 0.0027720, -0.0029458, 0.0032992
2: 0.0114147, 0.0177346, 0.0121886, 0.0175129, -0.0046865, 0.0041187
3: -0.0020435, 0.0027087, -0.0014616, 0.0025421, -0.0034138, 0.0029809
4: -0.0062646, -0.0018811, -0.0057278, -0.0020348, -0.0040551, 0.0038219
5: 0.0058984, 0.0106420, 0.0064792, 0.0104756, -0.0033974, 0.0029654
6: 0.0068906, 0.0107070, 0.0076743, 0.0104878, -0.0035972, 0.0030327
7: -0.0215020, -0.0112042, -0.0211408, -0.0124652, -0.0056676, 0.0065656
8: 0.9621851, 0.9916895, 0.9632200, 0.9880767, -0.0195267, 0.0221372
9: 0.0000956, 0.0087670, 0.0011574, 0.0084628, -0.0057258, 0.0049517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111336, upper bound: 0.0104720
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110829, upper bound: 0.0104720
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004705, 0.0013701, -0.0004439, 0.0009438, -0.0011122, 0.0014972
1: -0.0008822, 0.0034213, -0.0007580, 0.0027681, -0.0030667, 0.0033975
2: 0.0112162, 0.0176612, 0.0121945, 0.0174751, -0.0048401, 0.0042710
3: -0.0021928, 0.0026536, -0.0014571, 0.0025137, -0.0035298, 0.0030807
4: -0.0064022, -0.0019320, -0.0057237, -0.0020610, -0.0041526, 0.0037917
5: 0.0057494, 0.0105869, 0.0064837, 0.0104473, -0.0035133, 0.0030631
6: 0.0066896, 0.0107632, 0.0076803, 0.0104861, -0.0037965, 0.0030829
7: -0.0213824, -0.0108808, -0.0210793, -0.0124749, -0.0057514, 0.0068417
8: 0.9625279, 0.9926161, 0.9633963, 0.9880489, -0.0202513, 0.0228473
9: -0.0001767, 0.0086663, 0.0011656, 0.0084110, -0.0059522, 0.0050437

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 72

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110486, upper bound: 0.0102741
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109946, upper bound: 0.0102741
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004932, 0.0012584, -0.0004370, 0.0009720, -0.0011388, 0.0013748
1: -0.0009886, 0.0032502, -0.0007255, 0.0028113, -0.0029934, 0.0032371
2: 0.0114724, 0.0178205, 0.0121298, 0.0174265, -0.0045856, 0.0042193
3: -0.0020001, 0.0027734, -0.0015058, 0.0024771, -0.0033343, 0.0030631
4: -0.0062245, -0.0018215, -0.0057686, -0.0020947, -0.0040326, 0.0038383
5: 0.0059417, 0.0107065, 0.0064351, 0.0104108, -0.0033180, 0.0030474
6: 0.0069491, 0.0106907, 0.0076148, 0.0105045, -0.0035554, 0.0030759
7: -0.0216420, -0.0112983, -0.0210001, -0.0123694, -0.0058650, 0.0062932
8: 0.9617841, 0.9914199, 0.9636232, 0.9883512, -0.0199554, 0.0216697
9: 0.0001748, 0.0088849, 0.0010767, 0.0083443, -0.0055156, 0.0051213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.83 + 598.23 = 601.06 seconds
