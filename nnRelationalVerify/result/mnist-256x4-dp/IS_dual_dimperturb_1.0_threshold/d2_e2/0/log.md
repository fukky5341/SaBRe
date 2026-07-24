## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0018531149999999998


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000791, 0.0000791)
1: (-0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029634, 0.0029634)
2: (0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035563, 0.0035563)
3: (-0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0262303, 0.0262303)
4: (-0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019950)
5: (0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0020163)
6: (0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009807, 0.0009807)
7: (-0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0067978)
8: (0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0053931)
9: (0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0096999)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 2.02 = 3.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0022462, upper bound: 0.0022461

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021489, upper bound: 0.0020780
time: 1.03 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021495, upper bound: 0.0021494
time: 1.09 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.24 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.24
Output dim: 2, lower bound: -0.0021489, upper bound: 0.0020780
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.24
Output dim: 2, lower bound: -0.0021495, upper bound: 0.0021494

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0041905, -0.0041192, -0.0041910, -0.0041132, -0.0000773, 0.0000717
1: -0.0095893, -0.0069211, -0.0096077, -0.0066945, -0.0028948, 0.0026867
2: 0.9649559, 0.9681579, 0.9649338, 0.9684297, -0.0034738, 0.0032241
3: -0.0121734, 0.0114439, -0.0123365, 0.0134489, -0.0256224, 0.0237804
4: -0.0015634, 0.0002328, -0.0017159, 0.0002452, -0.0018086, 0.0019487
5: 0.0156902, 0.0175057, 0.0155361, 0.0175182, -0.0018280, 0.0019695
6: 0.0033920, 0.0042751, 0.0033859, 0.0043500, -0.0009580, 0.0008891
7: -0.0107440, -0.0046234, -0.0112637, -0.0045811, -0.0061629, 0.0066403
8: 0.0082053, 0.0130612, 0.0077931, 0.0130947, -0.0048894, 0.0052681
9: 0.0124827, 0.0212164, 0.0117412, 0.0212767, -0.0087940, 0.0094751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020777, upper bound: 0.0020777
time: 1.09 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020777, upper bound: 0.0020776
time: 1.75 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0041909, -0.0041145, -0.0041910, -0.0041119, -0.0000791, 0.0000732
1: -0.0096067, -0.0067430, -0.0096089, -0.0066455, -0.0029612, 0.0027421
2: 0.9649351, 0.9683716, 0.9649323, 0.9684886, -0.0035535, 0.0032906
3: -0.0123272, 0.0130198, -0.0123469, 0.0138834, -0.0262106, 0.0242712
4: -0.0016833, 0.0002445, -0.0017489, 0.0002460, -0.0018460, 0.0019935
5: 0.0155691, 0.0175175, 0.0155027, 0.0175190, -0.0018657, 0.0020147
6: 0.0033863, 0.0043340, 0.0033856, 0.0043663, -0.0009800, 0.0009075
7: -0.0111524, -0.0045835, -0.0113762, -0.0045784, -0.0062901, 0.0067927
8: 0.0078813, 0.0130928, 0.0077038, 0.0130968, -0.0049903, 0.0053890
9: 0.0118999, 0.0212732, 0.0115806, 0.0212805, -0.0089755, 0.0096926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020777, upper bound: 0.0021489
time: 1.04 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020777, upper bound: 0.0021494
time: 1.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.96 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 2, lower bound: -0.0020777, upper bound: 0.0020777
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 2, lower bound: -0.0020777, upper bound: 0.0020776
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 2, lower bound: -0.0020777, upper bound: 0.0021489
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 2, lower bound: -0.0020777, upper bound: 0.0021494

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041905, -0.0041192, -0.0041905, -0.0041192, -0.0000709, 0.0000709
1: -0.0095893, -0.0069211, -0.0095893, -0.0069211, -0.0026532, 0.0026532
2: 0.9649559, 0.9681579, 0.9649559, 0.9681579, -0.0031840, 0.0031840
3: -0.0121734, 0.0114439, -0.0121734, 0.0114439, -0.0234847, 0.0234847
4: -0.0015634, 0.0002328, -0.0015634, 0.0002328, -0.0017861, 0.0017861
5: 0.0156902, 0.0175057, 0.0156902, 0.0175057, -0.0018052, 0.0018052
6: 0.0033920, 0.0042751, 0.0033920, 0.0042751, -0.0008781, 0.0008781
7: -0.0107440, -0.0046234, -0.0107440, -0.0046234, -0.0060863, 0.0060863
8: 0.0082053, 0.0130612, 0.0082053, 0.0130612, -0.0048285, 0.0048285
9: 0.0124827, 0.0212164, 0.0124827, 0.0212164, -0.0086846, 0.0086846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 133

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019631, upper bound: 0.0019793
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019692, upper bound: 0.0019695
time: 1.80 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041905, -0.0041192, -0.0041909, -0.0041145, -0.0000760, 0.0000717
1: -0.0095893, -0.0069211, -0.0096067, -0.0067430, -0.0028463, 0.0026856
2: 0.9649559, 0.9681579, 0.9649351, 0.9683716, -0.0034156, 0.0032228
3: -0.0121734, 0.0114439, -0.0123272, 0.0130198, -0.0251932, 0.0237711
4: -0.0015634, 0.0002328, -0.0016833, 0.0002445, -0.0018079, 0.0019161
5: 0.0156902, 0.0175057, 0.0155691, 0.0175175, -0.0018272, 0.0019365
6: 0.0033920, 0.0042751, 0.0033863, 0.0043340, -0.0009419, 0.0008888
7: -0.0107440, -0.0046234, -0.0111524, -0.0045835, -0.0061605, 0.0065290
8: 0.0082053, 0.0130612, 0.0078813, 0.0130928, -0.0048874, 0.0051798
9: 0.0124827, 0.0212164, 0.0118999, 0.0212732, -0.0087905, 0.0093164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 133

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019631, upper bound: 0.0019793
time: 1.58 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019692, upper bound: 0.0019695
time: 1.06 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041909, -0.0041145, -0.0041905, -0.0041192, -0.0000717, 0.0000760
1: -0.0096067, -0.0067430, -0.0095893, -0.0069211, -0.0026856, 0.0028463
2: 0.9649351, 0.9683716, 0.9649559, 0.9681579, -0.0032228, 0.0034156
3: -0.0123272, 0.0130198, -0.0121734, 0.0114439, -0.0237711, 0.0251932
4: -0.0016833, 0.0002445, -0.0015634, 0.0002328, -0.0019161, 0.0018079
5: 0.0155691, 0.0175175, 0.0156902, 0.0175057, -0.0019365, 0.0018272
6: 0.0033863, 0.0043340, 0.0033920, 0.0042751, -0.0008888, 0.0009419
7: -0.0111524, -0.0045835, -0.0107440, -0.0046234, -0.0065290, 0.0061605
8: 0.0078813, 0.0130928, 0.0082053, 0.0130612, -0.0051798, 0.0048874
9: 0.0118999, 0.0212732, 0.0124827, 0.0212164, -0.0093164, 0.0087905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 133

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019789, upper bound: 0.0020365
time: 1.06 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019692, upper bound: 0.0020427
time: 1.67 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041909, -0.0041145, -0.0041909, -0.0041145, -0.0000732, 0.0000732
1: -0.0096067, -0.0067430, -0.0096067, -0.0067430, -0.0027401, 0.0027401
2: 0.9649351, 0.9683716, 0.9649351, 0.9683716, -0.0032882, 0.0032882
3: -0.0123272, 0.0130198, -0.0123272, 0.0130198, -0.0242532, 0.0242532
4: -0.0016833, 0.0002445, -0.0016833, 0.0002445, -0.0018446, 0.0018446
5: 0.0155691, 0.0175175, 0.0155691, 0.0175175, -0.0018643, 0.0018643
6: 0.0033863, 0.0043340, 0.0033863, 0.0043340, -0.0009068, 0.0009068
7: -0.0111524, -0.0045835, -0.0111524, -0.0045835, -0.0062854, 0.0062854
8: 0.0078813, 0.0130928, 0.0078813, 0.0130928, -0.0049866, 0.0049866
9: 0.0118999, 0.0212732, 0.0118999, 0.0212732, -0.0089688, 0.0089688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 133

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019631, upper bound: 0.0020485
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019692, upper bound: 0.0020434
time: 1.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.91 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.91
Output dim: 2, lower bound: -0.0019631, upper bound: 0.0019793
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.91
Output dim: 2, lower bound: -0.0019692, upper bound: 0.0019695
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.91
Output dim: 2, lower bound: -0.0019631, upper bound: 0.0019793
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.91
Output dim: 2, lower bound: -0.0019692, upper bound: 0.0019695
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.91
Output dim: 2, lower bound: -0.0019789, upper bound: 0.0020365
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.91
Output dim: 2, lower bound: -0.0019692, upper bound: 0.0020427
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.91
Output dim: 2, lower bound: -0.0019631, upper bound: 0.0020485
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.91
Output dim: 2, lower bound: -0.0019692, upper bound: 0.0020434

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041912, -0.0041236, -0.0041905, -0.0041198, -0.0000684, 0.0000653
1: -0.0096144, -0.0070831, -0.0095886, -0.0069408, -0.0025611, 0.0024441
2: 0.9649258, 0.9679635, 0.9649568, 0.9681342, -0.0030734, 0.0029330
3: -0.0123954, 0.0100097, -0.0121673, 0.0112696, -0.0226691, 0.0216336
4: -0.0014543, 0.0002497, -0.0015502, 0.0002324, -0.0016454, 0.0017241
5: 0.0158005, 0.0175227, 0.0157036, 0.0175052, -0.0016629, 0.0017425
6: 0.0033837, 0.0042214, 0.0033923, 0.0042685, -0.0008476, 0.0008088
7: -0.0103723, -0.0045659, -0.0106989, -0.0046250, -0.0056065, 0.0058749
8: 0.0085002, 0.0131068, 0.0082412, 0.0130599, -0.0044480, 0.0046609
9: 0.0130131, 0.0212984, 0.0125471, 0.0212141, -0.0080001, 0.0083830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 133

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019622, upper bound: 0.0019622
time: 1.46 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019622, upper bound: 0.0019621
time: 1.70 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041904, -0.0041231, -0.0041905, -0.0041196, -0.0000706, 0.0000645
1: -0.0095860, -0.0070661, -0.0095890, -0.0069331, -0.0026425, 0.0024143
2: 0.9649598, 0.9679838, 0.9649562, 0.9681435, -0.0031711, 0.0028973
3: -0.0121444, 0.0101601, -0.0121709, 0.0113377, -0.0233895, 0.0213696
4: -0.0014658, 0.0002306, -0.0015553, 0.0002326, -0.0016253, 0.0017789
5: 0.0157889, 0.0175034, 0.0156984, 0.0175055, -0.0016426, 0.0017979
6: 0.0033931, 0.0042271, 0.0033921, 0.0042711, -0.0008745, 0.0007990
7: -0.0104113, -0.0046309, -0.0107165, -0.0046240, -0.0055381, 0.0060616
8: 0.0084693, 0.0130552, 0.0082272, 0.0130606, -0.0043937, 0.0048090
9: 0.0129574, 0.0212056, 0.0125219, 0.0212154, -0.0079025, 0.0086494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 133

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019622, upper bound: 0.0019633
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019622, upper bound: 0.0019695
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041912, -0.0041236, -0.0041909, -0.0041150, -0.0000750, 0.0000674
1: -0.0096144, -0.0070831, -0.0096060, -0.0067627, -0.0028101, 0.0025229
2: 0.9649258, 0.9679635, 0.9649359, 0.9683480, -0.0033722, 0.0030276
3: -0.0123954, 0.0100097, -0.0123217, 0.0128460, -0.0248728, 0.0223314
4: -0.0014543, 0.0002497, -0.0016700, 0.0002441, -0.0016984, 0.0018917
5: 0.0158005, 0.0175227, 0.0155825, 0.0175171, -0.0017166, 0.0019119
6: 0.0033837, 0.0042214, 0.0033865, 0.0043275, -0.0009300, 0.0008349
7: -0.0103723, -0.0045659, -0.0111074, -0.0045850, -0.0057874, 0.0064460
8: 0.0085002, 0.0131068, 0.0079171, 0.0130916, -0.0045914, 0.0051140
9: 0.0130131, 0.0212984, 0.0119642, 0.0212712, -0.0082581, 0.0091979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 133

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020348, upper bound: 0.0019622
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020348, upper bound: 0.0019622
time: 1.65 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041904, -0.0041231, -0.0041909, -0.0041148, -0.0000756, 0.0000667
1: -0.0095860, -0.0070661, -0.0096064, -0.0067548, -0.0028312, 0.0024964
2: 0.9649598, 0.9679838, 0.9649354, 0.9683574, -0.0033976, 0.0029958
3: -0.0121444, 0.0101601, -0.0123252, 0.0129154, -0.0250598, 0.0220962
4: -0.0014658, 0.0002306, -0.0016753, 0.0002444, -0.0016805, 0.0019059
5: 0.0157889, 0.0175034, 0.0155771, 0.0175173, -0.0016985, 0.0019263
6: 0.0033931, 0.0042271, 0.0033864, 0.0043301, -0.0009369, 0.0008261
7: -0.0104113, -0.0046309, -0.0111254, -0.0045841, -0.0057264, 0.0064945
8: 0.0084693, 0.0130552, 0.0079028, 0.0130924, -0.0045431, 0.0051524
9: 0.0129574, 0.0212056, 0.0119385, 0.0212725, -0.0081712, 0.0092671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 133

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020348, upper bound: 0.0019633
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020348, upper bound: 0.0019694
time: 1.66 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0041909, -0.0041150, -0.0041912, -0.0041236, -0.0000674, 0.0000750
1: -0.0096060, -0.0067627, -0.0096144, -0.0070831, -0.0025229, 0.0028101
2: 0.9649359, 0.9683480, 0.9649258, 0.9679635, -0.0030276, 0.0033722
3: -0.0123217, 0.0128460, -0.0123954, 0.0100097, -0.0223314, 0.0248728
4: -0.0016700, 0.0002441, -0.0014543, 0.0002497, -0.0018917, 0.0016984
5: 0.0155825, 0.0175171, 0.0158005, 0.0175227, -0.0019119, 0.0017166
6: 0.0033865, 0.0043275, 0.0033837, 0.0042214, -0.0008349, 0.0009300
7: -0.0111074, -0.0045850, -0.0103723, -0.0045659, -0.0064460, 0.0057874
8: 0.0079171, 0.0130916, 0.0085002, 0.0131068, -0.0051140, 0.0045914
9: 0.0119642, 0.0212712, 0.0130131, 0.0212984, -0.0091979, 0.0082581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 133

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019622, upper bound: 0.0020348
time: 1.08 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019622, upper bound: 0.0020365
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0041909, -0.0041148, -0.0041904, -0.0041231, -0.0000667, 0.0000756
1: -0.0096064, -0.0067548, -0.0095860, -0.0070661, -0.0024964, 0.0028312
2: 0.9649354, 0.9683574, 0.9649598, 0.9679838, -0.0029958, 0.0033976
3: -0.0123252, 0.0129154, -0.0121444, 0.0101601, -0.0220962, 0.0250598
4: -0.0016753, 0.0002444, -0.0014658, 0.0002306, -0.0019059, 0.0016805
5: 0.0155771, 0.0175173, 0.0157889, 0.0175034, -0.0019263, 0.0016985
6: 0.0033864, 0.0043301, 0.0033931, 0.0042271, -0.0008261, 0.0009369
7: -0.0111254, -0.0045841, -0.0104113, -0.0046309, -0.0064945, 0.0057264
8: 0.0079028, 0.0130924, 0.0084693, 0.0130552, -0.0051524, 0.0045431
9: 0.0119385, 0.0212725, 0.0129574, 0.0212056, -0.0092671, 0.0081712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 133

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019633, upper bound: 0.0020348
time: 1.08 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019633, upper bound: 0.0020427
time: 1.54 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041915, -0.0041188, -0.0041909, -0.0041150, -0.0000707, 0.0000675
1: -0.0096264, -0.0069039, -0.0096060, -0.0067627, -0.0026484, 0.0025277
2: 0.9649115, 0.9681785, 0.9649359, 0.9683480, -0.0031781, 0.0030334
3: -0.0125017, 0.0115959, -0.0123217, 0.0128460, -0.0234414, 0.0223738
4: -0.0015750, 0.0002578, -0.0016700, 0.0002441, -0.0017017, 0.0017829
5: 0.0156786, 0.0175309, 0.0155825, 0.0175171, -0.0017198, 0.0018019
6: 0.0033798, 0.0042807, 0.0033865, 0.0043275, -0.0008764, 0.0008365
7: -0.0107834, -0.0045383, -0.0111074, -0.0045850, -0.0057984, 0.0060751
8: 0.0081741, 0.0131287, 0.0079171, 0.0130916, -0.0046002, 0.0048197
9: 0.0124265, 0.0213378, 0.0119642, 0.0212712, -0.0082738, 0.0086686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 133

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019668, upper bound: 0.0020352
time: 1.54 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019668, upper bound: 0.0020352
time: 1.65 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041909, -0.0041183, -0.0041909, -0.0041148, -0.0000729, 0.0000668
1: -0.0096039, -0.0068846, -0.0096064, -0.0067548, -0.0027294, 0.0025001
2: 0.9649384, 0.9682016, 0.9649354, 0.9683574, -0.0032754, 0.0030003
3: -0.0123026, 0.0117664, -0.0123252, 0.0129154, -0.0241591, 0.0221294
4: -0.0015879, 0.0002427, -0.0016753, 0.0002444, -0.0016831, 0.0018374
5: 0.0156655, 0.0175156, 0.0155771, 0.0175173, -0.0017010, 0.0018571
6: 0.0033872, 0.0042871, 0.0033864, 0.0043301, -0.0009033, 0.0008274
7: -0.0108276, -0.0045899, -0.0111254, -0.0045841, -0.0057350, 0.0062610
8: 0.0081390, 0.0130877, 0.0079028, 0.0130924, -0.0045499, 0.0049672
9: 0.0123634, 0.0212641, 0.0119385, 0.0212725, -0.0081834, 0.0089340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 133

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019668, upper bound: 0.0020368
time: 1.47 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019668, upper bound: 0.0020434
time: 1.05 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.82 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 2, lower bound: -0.0019622, upper bound: 0.0019622
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 2, lower bound: -0.0019622, upper bound: 0.0019621
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 2, lower bound: -0.0019622, upper bound: 0.0019633
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 2, lower bound: -0.0019622, upper bound: 0.0019695
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 2, lower bound: -0.0020348, upper bound: 0.0019622
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 2, lower bound: -0.0020348, upper bound: 0.0019622
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 2, lower bound: -0.0020348, upper bound: 0.0019633
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 2, lower bound: -0.0020348, upper bound: 0.0019694
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 2, lower bound: -0.0019622, upper bound: 0.0020348
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 2, lower bound: -0.0019622, upper bound: 0.0020365
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 2, lower bound: -0.0019633, upper bound: 0.0020348
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 2, lower bound: -0.0019633, upper bound: 0.0020427
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 2, lower bound: -0.0019668, upper bound: 0.0020352
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 2, lower bound: -0.0019668, upper bound: 0.0020352
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 2, lower bound: -0.0019668, upper bound: 0.0020368
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 2, lower bound: -0.0019668, upper bound: 0.0020434

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041912, -0.0041236, -0.0041912, -0.0041236, -0.0000635, 0.0000635
1: -0.0096144, -0.0070831, -0.0096144, -0.0070831, -0.0023785, 0.0023785
2: 0.9649258, 0.9679635, 0.9649258, 0.9679635, -0.0028543, 0.0028543
3: -0.0123954, 0.0100097, -0.0123954, 0.0100097, -0.0210532, 0.0210532
4: -0.0014543, 0.0002497, -0.0014543, 0.0002497, -0.0016012, 0.0016012
5: 0.0158005, 0.0175227, 0.0158005, 0.0175227, -0.0016183, 0.0016183
6: 0.0033837, 0.0042214, 0.0033837, 0.0042214, -0.0007871, 0.0007871
7: -0.0103723, -0.0045659, -0.0103723, -0.0045659, -0.0054561, 0.0054561
8: 0.0085002, 0.0131068, 0.0085002, 0.0131068, -0.0043286, 0.0043286
9: 0.0130131, 0.0212984, 0.0130131, 0.0212984, -0.0077854, 0.0077854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017917, upper bound: 0.0018306
time: 1.40 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0017625
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041912, -0.0041236, -0.0041904, -0.0041231, -0.0000666, 0.0000652
1: -0.0096144, -0.0070831, -0.0095860, -0.0070661, -0.0024950, 0.0024419
2: 0.9649258, 0.9679635, 0.9649598, 0.9679838, -0.0029941, 0.0029304
3: -0.0123954, 0.0100097, -0.0121444, 0.0101601, -0.0220839, 0.0216142
4: -0.0014543, 0.0002497, -0.0014658, 0.0002306, -0.0016439, 0.0016796
5: 0.0158005, 0.0175227, 0.0157889, 0.0175034, -0.0016614, 0.0016975
6: 0.0033837, 0.0042214, 0.0033931, 0.0042271, -0.0008257, 0.0008081
7: -0.0103723, -0.0045659, -0.0104113, -0.0046309, -0.0056015, 0.0057232
8: 0.0085002, 0.0131068, 0.0084693, 0.0130552, -0.0044440, 0.0045406
9: 0.0130131, 0.0212984, 0.0129574, 0.0212056, -0.0079929, 0.0081666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018143, upper bound: 0.0017917
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0017680
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041904, -0.0041231, -0.0041912, -0.0041236, -0.0000652, 0.0000666
1: -0.0095860, -0.0070661, -0.0096144, -0.0070831, -0.0024419, 0.0024950
2: 0.9649598, 0.9679838, 0.9649258, 0.9679635, -0.0029304, 0.0029941
3: -0.0121444, 0.0101601, -0.0123954, 0.0100097, -0.0216142, 0.0220839
4: -0.0014658, 0.0002306, -0.0014543, 0.0002497, -0.0016796, 0.0016439
5: 0.0157889, 0.0175034, 0.0158005, 0.0175227, -0.0016975, 0.0016614
6: 0.0033931, 0.0042271, 0.0033837, 0.0042214, -0.0008081, 0.0008257
7: -0.0104113, -0.0046309, -0.0103723, -0.0045659, -0.0057232, 0.0056015
8: 0.0084693, 0.0130552, 0.0085002, 0.0131068, -0.0045406, 0.0044440
9: 0.0129574, 0.0212056, 0.0130131, 0.0212984, -0.0081666, 0.0079929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017917, upper bound: 0.0018144
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017680, upper bound: 0.0017779
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041904, -0.0041231, -0.0041904, -0.0041231, -0.0000644, 0.0000644
1: -0.0095860, -0.0070661, -0.0095860, -0.0070661, -0.0024116, 0.0024116
2: 0.9649598, 0.9679838, 0.9649598, 0.9679838, -0.0028940, 0.0028940
3: -0.0121444, 0.0101601, -0.0121444, 0.0101601, -0.0213456, 0.0213456
4: -0.0014658, 0.0002306, -0.0014658, 0.0002306, -0.0016235, 0.0016235
5: 0.0157889, 0.0175034, 0.0157889, 0.0175034, -0.0016408, 0.0016408
6: 0.0033931, 0.0042271, 0.0033931, 0.0042271, -0.0007981, 0.0007981
7: -0.0104113, -0.0046309, -0.0104113, -0.0046309, -0.0055319, 0.0055319
8: 0.0084693, 0.0130552, 0.0084693, 0.0130552, -0.0043887, 0.0043887
9: 0.0129574, 0.0212056, 0.0129574, 0.0212056, -0.0078936, 0.0078936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017917, upper bound: 0.0018361
time: 1.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017680, upper bound: 0.0018097
time: 1.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041912, -0.0041236, -0.0041915, -0.0041188, -0.0000701, 0.0000657
1: -0.0096144, -0.0070831, -0.0096264, -0.0069039, -0.0026248, 0.0024610
2: 0.9649258, 0.9679635, 0.9649115, 0.9681785, -0.0031498, 0.0029533
3: -0.0123954, 0.0100097, -0.0125017, 0.0115959, -0.0232327, 0.0217829
4: -0.0014543, 0.0002497, -0.0015750, 0.0002578, -0.0016567, 0.0017670
5: 0.0158005, 0.0175227, 0.0156786, 0.0175309, -0.0016744, 0.0017858
6: 0.0033837, 0.0042214, 0.0033798, 0.0042807, -0.0008686, 0.0008144
7: -0.0103723, -0.0045659, -0.0107834, -0.0045383, -0.0056452, 0.0060210
8: 0.0085002, 0.0131068, 0.0081741, 0.0131287, -0.0044787, 0.0047767
9: 0.0130131, 0.0212984, 0.0124265, 0.0213378, -0.0080553, 0.0085914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018799, upper bound: 0.0018306
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018493, upper bound: 0.0017625
time: 1.42 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041912, -0.0041236, -0.0041909, -0.0041183, -0.0000729, 0.0000673
1: -0.0096144, -0.0070831, -0.0096039, -0.0068846, -0.0027297, 0.0025208
2: 0.9649258, 0.9679635, 0.9649384, 0.9682016, -0.0032758, 0.0030251
3: -0.0123954, 0.0100097, -0.0123026, 0.0117664, -0.0241618, 0.0223122
4: -0.0014543, 0.0002497, -0.0015879, 0.0002427, -0.0016970, 0.0018376
5: 0.0158005, 0.0175227, 0.0156655, 0.0175156, -0.0017151, 0.0018573
6: 0.0033837, 0.0042214, 0.0033872, 0.0042871, -0.0009034, 0.0008342
7: -0.0103723, -0.0045659, -0.0108276, -0.0045899, -0.0057824, 0.0062617
8: 0.0085002, 0.0131068, 0.0081390, 0.0130877, -0.0045875, 0.0049678
9: 0.0130131, 0.0212984, 0.0123634, 0.0212641, -0.0082510, 0.0089350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018799, upper bound: 0.0018307
time: 1.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018493, upper bound: 0.0017680
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041904, -0.0041231, -0.0041915, -0.0041188, -0.0000716, 0.0000684
1: -0.0095860, -0.0070661, -0.0096264, -0.0069039, -0.0026821, 0.0025603
2: 0.9649598, 0.9679838, 0.9649115, 0.9681785, -0.0032187, 0.0030724
3: -0.0121444, 0.0101601, -0.0125017, 0.0115959, -0.0237403, 0.0226618
4: -0.0014658, 0.0002306, -0.0015750, 0.0002578, -0.0017236, 0.0018056
5: 0.0157889, 0.0175034, 0.0156786, 0.0175309, -0.0017420, 0.0018249
6: 0.0033931, 0.0042271, 0.0033798, 0.0042807, -0.0008876, 0.0008473
7: -0.0104113, -0.0046309, -0.0107834, -0.0045383, -0.0058730, 0.0061525
8: 0.0084693, 0.0130552, 0.0081741, 0.0131287, -0.0046594, 0.0048811
9: 0.0129574, 0.0212056, 0.0124265, 0.0213378, -0.0083803, 0.0087791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018799, upper bound: 0.0018144
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018539, upper bound: 0.0017779
time: 1.44 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041904, -0.0041231, -0.0041909, -0.0041183, -0.0000709, 0.0000666
1: -0.0095860, -0.0070661, -0.0096039, -0.0068846, -0.0026558, 0.0024943
2: 0.9649598, 0.9679838, 0.9649384, 0.9682016, -0.0031871, 0.0029933
3: -0.0121444, 0.0101601, -0.0123026, 0.0117664, -0.0235074, 0.0220779
4: -0.0014658, 0.0002306, -0.0015879, 0.0002427, -0.0016791, 0.0017879
5: 0.0157889, 0.0175034, 0.0156655, 0.0175156, -0.0016971, 0.0018070
6: 0.0033931, 0.0042271, 0.0033872, 0.0042871, -0.0008789, 0.0008255
7: -0.0104113, -0.0046309, -0.0108276, -0.0045899, -0.0057217, 0.0060921
8: 0.0084693, 0.0130552, 0.0081390, 0.0130877, -0.0045393, 0.0048332
9: 0.0129574, 0.0212056, 0.0123634, 0.0212641, -0.0081644, 0.0086930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018799, upper bound: 0.0018361
time: 1.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018539, upper bound: 0.0018097
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041915, -0.0041188, -0.0041912, -0.0041236, -0.0000657, 0.0000701
1: -0.0096264, -0.0069039, -0.0096144, -0.0070831, -0.0024610, 0.0026248
2: 0.9649115, 0.9681785, 0.9649258, 0.9679635, -0.0029533, 0.0031498
3: -0.0125017, 0.0115959, -0.0123954, 0.0100097, -0.0217829, 0.0232327
4: -0.0015750, 0.0002578, -0.0014543, 0.0002497, -0.0017670, 0.0016567
5: 0.0156786, 0.0175309, 0.0158005, 0.0175227, -0.0017858, 0.0016744
6: 0.0033798, 0.0042807, 0.0033837, 0.0042214, -0.0008144, 0.0008686
7: -0.0107834, -0.0045383, -0.0103723, -0.0045659, -0.0060210, 0.0056452
8: 0.0081741, 0.0131287, 0.0085002, 0.0131068, -0.0047767, 0.0044787
9: 0.0124265, 0.0213378, 0.0130131, 0.0212984, -0.0085914, 0.0080553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018307, upper bound: 0.0018799
time: 1.42 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0018493
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041909, -0.0041183, -0.0041912, -0.0041236, -0.0000673, 0.0000729
1: -0.0096039, -0.0068846, -0.0096144, -0.0070831, -0.0025208, 0.0027297
2: 0.9649384, 0.9682016, 0.9649258, 0.9679635, -0.0030251, 0.0032758
3: -0.0123026, 0.0117664, -0.0123954, 0.0100097, -0.0223122, 0.0241618
4: -0.0015879, 0.0002427, -0.0014543, 0.0002497, -0.0018376, 0.0016970
5: 0.0156655, 0.0175156, 0.0158005, 0.0175227, -0.0018573, 0.0017151
6: 0.0033872, 0.0042871, 0.0033837, 0.0042214, -0.0008342, 0.0009034
7: -0.0108276, -0.0045899, -0.0103723, -0.0045659, -0.0062617, 0.0057824
8: 0.0081390, 0.0130877, 0.0085002, 0.0131068, -0.0049678, 0.0045875
9: 0.0123634, 0.0212641, 0.0130131, 0.0212984, -0.0089350, 0.0082510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018307, upper bound: 0.0019003
time: 1.08 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0018622
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041915, -0.0041188, -0.0041904, -0.0041231, -0.0000684, 0.0000716
1: -0.0096264, -0.0069039, -0.0095860, -0.0070661, -0.0025603, 0.0026821
2: 0.9649115, 0.9681785, 0.9649598, 0.9679838, -0.0030724, 0.0032187
3: -0.0125017, 0.0115959, -0.0121444, 0.0101601, -0.0226618, 0.0237403
4: -0.0015750, 0.0002578, -0.0014658, 0.0002306, -0.0018056, 0.0017236
5: 0.0156786, 0.0175309, 0.0157889, 0.0175034, -0.0018249, 0.0017420
6: 0.0033798, 0.0042807, 0.0033931, 0.0042271, -0.0008473, 0.0008876
7: -0.0107834, -0.0045383, -0.0104113, -0.0046309, -0.0061525, 0.0058730
8: 0.0081741, 0.0131287, 0.0084693, 0.0130552, -0.0048811, 0.0046594
9: 0.0124265, 0.0213378, 0.0129574, 0.0212056, -0.0087791, 0.0083803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018143, upper bound: 0.0018799
time: 1.88 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0018539
time: 1.52 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041909, -0.0041183, -0.0041904, -0.0041231, -0.0000666, 0.0000709
1: -0.0096039, -0.0068846, -0.0095860, -0.0070661, -0.0024943, 0.0026558
2: 0.9649384, 0.9682016, 0.9649598, 0.9679838, -0.0029933, 0.0031871
3: -0.0123026, 0.0117664, -0.0121444, 0.0101601, -0.0220779, 0.0235074
4: -0.0015879, 0.0002427, -0.0014658, 0.0002306, -0.0017879, 0.0016791
5: 0.0156655, 0.0175156, 0.0157889, 0.0175034, -0.0018070, 0.0016971
6: 0.0033872, 0.0042871, 0.0033931, 0.0042271, -0.0008255, 0.0008789
7: -0.0108276, -0.0045899, -0.0104113, -0.0046309, -0.0060921, 0.0057217
8: 0.0081390, 0.0130877, 0.0084693, 0.0130552, -0.0048332, 0.0045393
9: 0.0123634, 0.0212641, 0.0129574, 0.0212056, -0.0086930, 0.0081644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018143, upper bound: 0.0019110
time: 1.54 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0018926
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041915, -0.0041188, -0.0041915, -0.0041188, -0.0000658, 0.0000658
1: -0.0096264, -0.0069039, -0.0096264, -0.0069039, -0.0024625, 0.0024625
2: 0.9649115, 0.9681785, 0.9649115, 0.9681785, -0.0029551, 0.0029551
3: -0.0125017, 0.0115959, -0.0125017, 0.0115959, -0.0217966, 0.0217966
4: -0.0015750, 0.0002578, -0.0015750, 0.0002578, -0.0016578, 0.0016578
5: 0.0156786, 0.0175309, 0.0156786, 0.0175309, -0.0016755, 0.0016755
6: 0.0033798, 0.0042807, 0.0033798, 0.0042807, -0.0008149, 0.0008149
7: -0.0107834, -0.0045383, -0.0107834, -0.0045383, -0.0056488, 0.0056488
8: 0.0081741, 0.0131287, 0.0081741, 0.0131287, -0.0044815, 0.0044815
9: 0.0124265, 0.0213378, 0.0124265, 0.0213378, -0.0080604, 0.0080604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018121, upper bound: 0.0019033
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017936, upper bound: 0.0018576
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041915, -0.0041188, -0.0041909, -0.0041183, -0.0000688, 0.0000674
1: -0.0096264, -0.0069039, -0.0096039, -0.0068846, -0.0025766, 0.0025257
2: 0.9649115, 0.9681785, 0.9649384, 0.9682016, -0.0030920, 0.0030310
3: -0.0125017, 0.0115959, -0.0123026, 0.0117664, -0.0228062, 0.0223558
4: -0.0015750, 0.0002578, -0.0015879, 0.0002427, -0.0017003, 0.0017345
5: 0.0156786, 0.0175309, 0.0156655, 0.0175156, -0.0017184, 0.0017531
6: 0.0033798, 0.0042807, 0.0033872, 0.0042871, -0.0008527, 0.0008358
7: -0.0107834, -0.0045383, -0.0108276, -0.0045899, -0.0057937, 0.0059104
8: 0.0081741, 0.0131287, 0.0081390, 0.0130877, -0.0045965, 0.0046891
9: 0.0124265, 0.0213378, 0.0123634, 0.0212641, -0.0082672, 0.0084337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018308, upper bound: 0.0018843
time: 1.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017936, upper bound: 0.0018622
time: 1.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041909, -0.0041183, -0.0041915, -0.0041188, -0.0000674, 0.0000688
1: -0.0096039, -0.0068846, -0.0096264, -0.0069039, -0.0025257, 0.0025766
2: 0.9649384, 0.9682016, 0.9649115, 0.9681785, -0.0030310, 0.0030920
3: -0.0123026, 0.0117664, -0.0125017, 0.0115959, -0.0223559, 0.0228062
4: -0.0015879, 0.0002427, -0.0015750, 0.0002578, -0.0017345, 0.0017003
5: 0.0156655, 0.0175156, 0.0156786, 0.0175309, -0.0017531, 0.0017184
6: 0.0033872, 0.0042871, 0.0033798, 0.0042807, -0.0008358, 0.0008527
7: -0.0108276, -0.0045899, -0.0107834, -0.0045383, -0.0059104, 0.0057937
8: 0.0081390, 0.0130877, 0.0081741, 0.0131287, -0.0046891, 0.0045965
9: 0.0123634, 0.0212641, 0.0124265, 0.0213378, -0.0084337, 0.0082672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018121, upper bound: 0.0018905
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017989, upper bound: 0.0018708
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041909, -0.0041183, -0.0041909, -0.0041183, -0.0000667, 0.0000667
1: -0.0096039, -0.0068846, -0.0096039, -0.0068846, -0.0024975, 0.0024975
2: 0.9649384, 0.9682016, 0.9649384, 0.9682016, -0.0029972, 0.0029972
3: -0.0123026, 0.0117664, -0.0123026, 0.0117664, -0.0221066, 0.0221066
4: -0.0015879, 0.0002427, -0.0015879, 0.0002427, -0.0016813, 0.0016813
5: 0.0156655, 0.0175156, 0.0156655, 0.0175156, -0.0016993, 0.0016993
6: 0.0033872, 0.0042871, 0.0033872, 0.0042871, -0.0008265, 0.0008265
7: -0.0108276, -0.0045899, -0.0108276, -0.0045899, -0.0057291, 0.0057291
8: 0.0081390, 0.0130877, 0.0081390, 0.0130877, -0.0045452, 0.0045452
9: 0.0123634, 0.0212641, 0.0123634, 0.0212641, -0.0081750, 0.0081750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018121, upper bound: 0.0019145
time: 1.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017989, upper bound: 0.0019019
time: 1.61 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.47 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0017917, upper bound: 0.0018306
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0017625
IS_A1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0018143, upper bound: 0.0017917
IS_A1_B1_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0017680
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0017917, upper bound: 0.0018144
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0017680, upper bound: 0.0017779
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0017917, upper bound: 0.0018361
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0017680, upper bound: 0.0018097
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0018799, upper bound: 0.0018306
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0018493, upper bound: 0.0017625
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0018799, upper bound: 0.0018307
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0018493, upper bound: 0.0017680
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0018799, upper bound: 0.0018144
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0018539, upper bound: 0.0017779
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0018799, upper bound: 0.0018361
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0018539, upper bound: 0.0018097
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0018307, upper bound: 0.0018799
IS_A2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0018493
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0018307, upper bound: 0.0019003
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0018622
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0018143, upper bound: 0.0018799
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0018539
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0018143, upper bound: 0.0019110
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0018926
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0018121, upper bound: 0.0019033
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0017936, upper bound: 0.0018576
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0018308, upper bound: 0.0018843
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0017936, upper bound: 0.0018622
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0018121, upper bound: 0.0018905
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0017989, upper bound: 0.0018708
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0018121, upper bound: 0.0019145
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 2, lower bound: -0.0017989, upper bound: 0.0019019

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041895, -0.0041238, -0.0041915, -0.0041188, -0.0000687, 0.0000653
1: -0.0095541, -0.0070933, -0.0096264, -0.0069039, -0.0025711, 0.0024436
2: 0.9649981, 0.9679512, 0.9649115, 0.9681785, -0.0030854, 0.0029324
3: -0.0118616, 0.0099198, -0.0125017, 0.0115959, -0.0227574, 0.0216290
4: -0.0014475, 0.0002091, -0.0015750, 0.0002578, -0.0016450, 0.0017308
5: 0.0158074, 0.0174817, 0.0156786, 0.0175309, -0.0016626, 0.0017493
6: 0.0034037, 0.0042181, 0.0033798, 0.0042807, -0.0008509, 0.0008087
7: -0.0103491, -0.0047042, -0.0107834, -0.0045383, -0.0056053, 0.0058978
8: 0.0085187, 0.0129970, 0.0081741, 0.0131287, -0.0044470, 0.0046790
9: 0.0130463, 0.0211011, 0.0124265, 0.0213378, -0.0079984, 0.0084157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018494, upper bound: 0.0017625
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018494, upper bound: 0.0017625
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041895, -0.0041238, -0.0041909, -0.0041183, -0.0000713, 0.0000669
1: -0.0095541, -0.0070933, -0.0096039, -0.0068846, -0.0026694, 0.0025055
2: 0.9649981, 0.9679512, 0.9649384, 0.9682016, -0.0032034, 0.0030068
3: -0.0118616, 0.0099198, -0.0123026, 0.0117664, -0.0236280, 0.0221773
4: -0.0014475, 0.0002091, -0.0015879, 0.0002427, -0.0016867, 0.0017970
5: 0.0158074, 0.0174817, 0.0156655, 0.0175156, -0.0017047, 0.0018162
6: 0.0034037, 0.0042181, 0.0033872, 0.0042871, -0.0008834, 0.0008292
7: -0.0103491, -0.0047042, -0.0108276, -0.0045899, -0.0057474, 0.0061234
8: 0.0085187, 0.0129970, 0.0081390, 0.0130877, -0.0045597, 0.0048580
9: 0.0130463, 0.0211011, 0.0123634, 0.0212641, -0.0082011, 0.0087376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018622, upper bound: 0.0017680
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018622, upper bound: 0.0017680
time: 1.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041889, -0.0041234, -0.0041915, -0.0041188, -0.0000702, 0.0000680
1: -0.0095312, -0.0070782, -0.0096264, -0.0069039, -0.0026273, 0.0025481
2: 0.9650257, 0.9679694, 0.9649115, 0.9681785, -0.0031528, 0.0030579
3: -0.0116592, 0.0100526, -0.0125017, 0.0115959, -0.0232551, 0.0225544
4: -0.0014576, 0.0001937, -0.0015750, 0.0002578, -0.0017154, 0.0017687
5: 0.0157972, 0.0174661, 0.0156786, 0.0175309, -0.0017337, 0.0017876
6: 0.0034113, 0.0042230, 0.0033798, 0.0042807, -0.0008695, 0.0008433
7: -0.0103835, -0.0047567, -0.0107834, -0.0045383, -0.0058452, 0.0060268
8: 0.0084914, 0.0129554, 0.0081741, 0.0131287, -0.0046373, 0.0047813
9: 0.0129972, 0.0210262, 0.0124265, 0.0213378, -0.0083406, 0.0085997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018539, upper bound: 0.0017779
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018539, upper bound: 0.0017779
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041853, -0.0041170, -0.0041906, -0.0041190, -0.0000663, 0.0000726
1: -0.0093955, -0.0068383, -0.0095948, -0.0069112, -0.0024843, 0.0027186
2: 0.9651884, 0.9682573, 0.9649493, 0.9681697, -0.0029813, 0.0032624
3: -0.0104582, 0.0121768, -0.0122222, 0.0115314, -0.0219896, 0.0240632
4: -0.0016191, 0.0001024, -0.0015701, 0.0002365, -0.0018301, 0.0016724
5: 0.0156339, 0.0173738, 0.0156835, 0.0175094, -0.0018497, 0.0016903
6: 0.0034562, 0.0043025, 0.0033902, 0.0042783, -0.0008222, 0.0008997
7: -0.0109340, -0.0050679, -0.0107667, -0.0046108, -0.0062362, 0.0056988
8: 0.0080546, 0.0127085, 0.0081873, 0.0130712, -0.0049475, 0.0045212
9: 0.0122117, 0.0205821, 0.0124503, 0.0212344, -0.0088986, 0.0081317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018539, upper bound: 0.0017779
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018539, upper bound: 0.0017779
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041889, -0.0041234, -0.0041909, -0.0041183, -0.0000696, 0.0000661
1: -0.0095312, -0.0070782, -0.0096039, -0.0068846, -0.0026047, 0.0024742
2: 0.9650257, 0.9679694, 0.9649384, 0.9682016, -0.0031258, 0.0029691
3: -0.0116592, 0.0100526, -0.0123026, 0.0117664, -0.0230552, 0.0218999
4: -0.0014576, 0.0001937, -0.0015879, 0.0002427, -0.0016656, 0.0017535
5: 0.0157972, 0.0174661, 0.0156655, 0.0175156, -0.0016834, 0.0017722
6: 0.0034113, 0.0042230, 0.0033872, 0.0042871, -0.0008620, 0.0008188
7: -0.0103835, -0.0047567, -0.0108276, -0.0045899, -0.0056756, 0.0059750
8: 0.0084914, 0.0129554, 0.0081390, 0.0130877, -0.0045027, 0.0047403
9: 0.0129972, 0.0210262, 0.0123634, 0.0212641, -0.0080986, 0.0085258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018910, upper bound: 0.0018087
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018910, upper bound: 0.0018097
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041853, -0.0041170, -0.0041899, -0.0041184, -0.0000669, 0.0000713
1: -0.0093955, -0.0068383, -0.0095683, -0.0068914, -0.0025041, 0.0026711
2: 0.9651884, 0.9682573, 0.9649811, 0.9681935, -0.0030050, 0.0032055
3: -0.0104582, 0.0121768, -0.0119873, 0.0117065, -0.0221647, 0.0236429
4: -0.0016191, 0.0001024, -0.0015834, 0.0002187, -0.0017982, 0.0016858
5: 0.0156339, 0.0173738, 0.0156701, 0.0174914, -0.0018174, 0.0017037
6: 0.0034562, 0.0043025, 0.0033990, 0.0042849, -0.0008287, 0.0008840
7: -0.0109340, -0.0050679, -0.0108121, -0.0046716, -0.0061273, 0.0057442
8: 0.0080546, 0.0127085, 0.0081513, 0.0130229, -0.0048611, 0.0045571
9: 0.0122117, 0.0205821, 0.0123856, 0.0211475, -0.0087431, 0.0081965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018920, upper bound: 0.0018087
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018920, upper bound: 0.0018097
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041915, -0.0041188, -0.0041895, -0.0041238, -0.0000653, 0.0000687
1: -0.0096264, -0.0069039, -0.0095541, -0.0070933, -0.0024436, 0.0025711
2: 0.9649115, 0.9681785, 0.9649981, 0.9679512, -0.0029324, 0.0030854
3: -0.0125017, 0.0115959, -0.0118616, 0.0099198, -0.0216290, 0.0227574
4: -0.0015750, 0.0002578, -0.0014475, 0.0002091, -0.0017308, 0.0016450
5: 0.0156786, 0.0175309, 0.0158074, 0.0174817, -0.0017493, 0.0016626
6: 0.0033798, 0.0042807, 0.0034037, 0.0042181, -0.0008087, 0.0008509
7: -0.0107834, -0.0045383, -0.0103491, -0.0047042, -0.0058978, 0.0056053
8: 0.0081741, 0.0131287, 0.0085187, 0.0129970, -0.0046790, 0.0044470
9: 0.0124265, 0.0213378, 0.0130463, 0.0211011, -0.0084157, 0.0079984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0018494
time: 1.05 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0018494
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041909, -0.0041183, -0.0041895, -0.0041238, -0.0000669, 0.0000713
1: -0.0096039, -0.0068846, -0.0095541, -0.0070933, -0.0025055, 0.0026694
2: 0.9649384, 0.9682016, 0.9649981, 0.9679512, -0.0030068, 0.0032034
3: -0.0123026, 0.0117664, -0.0118616, 0.0099198, -0.0221773, 0.0236280
4: -0.0015879, 0.0002427, -0.0014475, 0.0002091, -0.0017970, 0.0016867
5: 0.0156655, 0.0175156, 0.0158074, 0.0174817, -0.0018162, 0.0017047
6: 0.0033872, 0.0042871, 0.0034037, 0.0042181, -0.0008292, 0.0008834
7: -0.0108276, -0.0045899, -0.0103491, -0.0047042, -0.0061234, 0.0057474
8: 0.0081390, 0.0130877, 0.0085187, 0.0129970, -0.0048580, 0.0045597
9: 0.0123634, 0.0212641, 0.0130463, 0.0211011, -0.0087376, 0.0082011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017680, upper bound: 0.0018622
time: 1.03 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017680, upper bound: 0.0018622
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041899, -0.0041184, -0.0041866, -0.0041166, -0.0000725, 0.0000681
1: -0.0095683, -0.0068914, -0.0094426, -0.0068234, -0.0027166, 0.0025512
2: 0.9649811, 0.9681935, 0.9651320, 0.9682752, -0.0032600, 0.0030615
3: -0.0119873, 0.0117065, -0.0108748, 0.0123087, -0.0240454, 0.0225813
4: -0.0015834, 0.0002187, -0.0016292, 0.0001341, -0.0017174, 0.0018288
5: 0.0156701, 0.0174914, 0.0156238, 0.0174058, -0.0017358, 0.0018483
6: 0.0033990, 0.0042849, 0.0034406, 0.0043074, -0.0008990, 0.0008443
7: -0.0108121, -0.0046716, -0.0109682, -0.0049600, -0.0058521, 0.0062316
8: 0.0081513, 0.0130229, 0.0080275, 0.0127942, -0.0046428, 0.0049438
9: 0.0123856, 0.0211475, 0.0121629, 0.0207361, -0.0083505, 0.0088920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017680, upper bound: 0.0018622
time: 1.03 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017680, upper bound: 0.0018622
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041915, -0.0041188, -0.0041889, -0.0041234, -0.0000680, 0.0000702
1: -0.0096264, -0.0069039, -0.0095312, -0.0070782, -0.0025481, 0.0026273
2: 0.9649115, 0.9681785, 0.9650257, 0.9679694, -0.0030579, 0.0031528
3: -0.0125017, 0.0115959, -0.0116592, 0.0100526, -0.0225544, 0.0232551
4: -0.0015750, 0.0002578, -0.0014576, 0.0001937, -0.0017687, 0.0017154
5: 0.0156786, 0.0175309, 0.0157972, 0.0174661, -0.0017876, 0.0017337
6: 0.0033798, 0.0042807, 0.0034113, 0.0042230, -0.0008433, 0.0008695
7: -0.0107834, -0.0045383, -0.0103835, -0.0047567, -0.0060268, 0.0058452
8: 0.0081741, 0.0131287, 0.0084914, 0.0129554, -0.0047813, 0.0046373
9: 0.0124265, 0.0213378, 0.0129972, 0.0210262, -0.0085997, 0.0083406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017779, upper bound: 0.0018539
time: 1.04 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017779, upper bound: 0.0018539
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041906, -0.0041190, -0.0041853, -0.0041170, -0.0000726, 0.0000663
1: -0.0095948, -0.0069112, -0.0093955, -0.0068383, -0.0027186, 0.0024843
2: 0.9649493, 0.9681697, 0.9651884, 0.9682573, -0.0032624, 0.0029813
3: -0.0122222, 0.0115314, -0.0104582, 0.0121768, -0.0240632, 0.0219896
4: -0.0015701, 0.0002365, -0.0016191, 0.0001024, -0.0016724, 0.0018301
5: 0.0156835, 0.0175094, 0.0156339, 0.0173738, -0.0016903, 0.0018497
6: 0.0033902, 0.0042783, 0.0034562, 0.0043025, -0.0008997, 0.0008222
7: -0.0107667, -0.0046108, -0.0109340, -0.0050679, -0.0056988, 0.0062362
8: 0.0081873, 0.0130712, 0.0080546, 0.0127085, -0.0045212, 0.0049475
9: 0.0124503, 0.0212344, 0.0122117, 0.0205821, -0.0081317, 0.0088985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017779, upper bound: 0.0018539
time: 1.02 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017779, upper bound: 0.0018539
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041909, -0.0041183, -0.0041889, -0.0041234, -0.0000661, 0.0000696
1: -0.0096039, -0.0068846, -0.0095312, -0.0070782, -0.0024742, 0.0026047
2: 0.9649384, 0.9682016, 0.9650257, 0.9679694, -0.0029691, 0.0031258
3: -0.0123026, 0.0117664, -0.0116592, 0.0100526, -0.0218999, 0.0230552
4: -0.0015879, 0.0002427, -0.0014576, 0.0001937, -0.0017535, 0.0016656
5: 0.0156655, 0.0175156, 0.0157972, 0.0174661, -0.0017722, 0.0016834
6: 0.0033872, 0.0042871, 0.0034113, 0.0042230, -0.0008188, 0.0008620
7: -0.0108276, -0.0045899, -0.0103835, -0.0047567, -0.0059750, 0.0056756
8: 0.0081390, 0.0130877, 0.0084914, 0.0129554, -0.0047403, 0.0045027
9: 0.0123634, 0.0212641, 0.0129972, 0.0210262, -0.0085258, 0.0080986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018078, upper bound: 0.0018919
time: 1.07 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018078, upper bound: 0.0018920
time: 1.51 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041899, -0.0041184, -0.0041853, -0.0041170, -0.0000713, 0.0000669
1: -0.0095683, -0.0068914, -0.0093955, -0.0068383, -0.0026711, 0.0025041
2: 0.9649811, 0.9681935, 0.9651884, 0.9682573, -0.0032055, 0.0030050
3: -0.0119873, 0.0117065, -0.0104582, 0.0121768, -0.0236429, 0.0221647
4: -0.0015834, 0.0002187, -0.0016191, 0.0001024, -0.0016858, 0.0017982
5: 0.0156701, 0.0174914, 0.0156339, 0.0173738, -0.0017037, 0.0018174
6: 0.0033990, 0.0042849, 0.0034562, 0.0043025, -0.0008840, 0.0008287
7: -0.0108121, -0.0046716, -0.0109340, -0.0050679, -0.0057442, 0.0061273
8: 0.0081513, 0.0130229, 0.0080546, 0.0127085, -0.0045571, 0.0048611
9: 0.0123856, 0.0211475, 0.0122117, 0.0205821, -0.0081965, 0.0087431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018078, upper bound: 0.0018926
time: 1.07 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018078, upper bound: 0.0018927
time: 1.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041899, -0.0041191, -0.0041915, -0.0041188, -0.0000643, 0.0000653
1: -0.0095685, -0.0069154, -0.0096264, -0.0069039, -0.0024091, 0.0024435
2: 0.9649808, 0.9681646, 0.9649115, 0.9681785, -0.0028910, 0.0029323
3: -0.0119893, 0.0114937, -0.0125017, 0.0115959, -0.0213233, 0.0216282
4: -0.0015672, 0.0002188, -0.0015750, 0.0002578, -0.0016450, 0.0016218
5: 0.0156864, 0.0174915, 0.0156786, 0.0175309, -0.0016625, 0.0016391
6: 0.0033989, 0.0042769, 0.0033798, 0.0042807, -0.0007972, 0.0008086
7: -0.0107569, -0.0046711, -0.0107834, -0.0045383, -0.0056051, 0.0055261
8: 0.0081951, 0.0130233, 0.0081741, 0.0131287, -0.0044469, 0.0043842
9: 0.0124643, 0.0211483, 0.0124265, 0.0213378, -0.0079981, 0.0078853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017940, upper bound: 0.0018576
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017940, upper bound: 0.0018576
time: 1.41 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041864, -0.0041129, -0.0041906, -0.0041190, -0.0000664, 0.0000709
1: -0.0094383, -0.0066830, -0.0095948, -0.0069112, -0.0024865, 0.0026536
2: 0.9651371, 0.9684435, 0.9649493, 0.9681697, -0.0029839, 0.0031844
3: -0.0108370, 0.0135508, -0.0122222, 0.0115314, -0.0220085, 0.0234877
4: -0.0017236, 0.0001312, -0.0015701, 0.0002365, -0.0017864, 0.0016739
5: 0.0155283, 0.0174029, 0.0156835, 0.0175094, -0.0018054, 0.0016917
6: 0.0034420, 0.0043538, 0.0033902, 0.0042783, -0.0008229, 0.0008782
7: -0.0112900, -0.0049697, -0.0107667, -0.0046108, -0.0060870, 0.0057037
8: 0.0077722, 0.0127864, 0.0081873, 0.0130712, -0.0048292, 0.0045250
9: 0.0117036, 0.0207222, 0.0124503, 0.0212344, -0.0086857, 0.0081387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017940, upper bound: 0.0018576
time: 1.41 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017940, upper bound: 0.0018576
time: 1.39 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0041915, -0.0041188, -0.0041895, -0.0041186, -0.0000682, 0.0000662
1: -0.0096264, -0.0069039, -0.0095507, -0.0068968, -0.0025558, 0.0024793
2: 0.9649115, 0.9681785, 0.9650022, 0.9681869, -0.0030670, 0.0029752
3: -0.0125017, 0.0115959, -0.0118319, 0.0116583, -0.0226219, 0.0219447
4: -0.0015750, 0.0002578, -0.0015797, 0.0002069, -0.0016690, 0.0017205
5: 0.0156786, 0.0175309, 0.0156738, 0.0174794, -0.0016868, 0.0017389
6: 0.0033798, 0.0042807, 0.0034048, 0.0042831, -0.0008458, 0.0008205
7: -0.0107834, -0.0045383, -0.0107996, -0.0047119, -0.0056872, 0.0058627
8: 0.0081741, 0.0131287, 0.0081613, 0.0129909, -0.0045119, 0.0046512
9: 0.0124265, 0.0213378, 0.0124034, 0.0210901, -0.0081151, 0.0083656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018068, upper bound: 0.0018621
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018068, upper bound: 0.0018621
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0041906, -0.0041190, -0.0041854, -0.0041131, -0.0000723, 0.0000665
1: -0.0095948, -0.0069112, -0.0094000, -0.0066906, -0.0027091, 0.0024888
2: 0.9649493, 0.9681697, 0.9651830, 0.9684344, -0.0032510, 0.0029867
3: -0.0122222, 0.0115314, -0.0104978, 0.0134834, -0.0239788, 0.0220292
4: -0.0015701, 0.0002365, -0.0017185, 0.0001054, -0.0016755, 0.0018237
5: 0.0156835, 0.0175094, 0.0155335, 0.0173769, -0.0016933, 0.0018432
6: 0.0033902, 0.0042783, 0.0034547, 0.0043513, -0.0008965, 0.0008236
7: -0.0107667, -0.0046108, -0.0112726, -0.0050577, -0.0057091, 0.0062143
8: 0.0081873, 0.0130712, 0.0077860, 0.0127166, -0.0045293, 0.0049302
9: 0.0124503, 0.0212344, 0.0117285, 0.0205967, -0.0081464, 0.0088674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018068, upper bound: 0.0018622
time: 1.17 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018068, upper bound: 0.0018622
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041895, -0.0041186, -0.0041915, -0.0041188, -0.0000662, 0.0000682
1: -0.0095507, -0.0068968, -0.0096264, -0.0069039, -0.0024793, 0.0025558
2: 0.9650022, 0.9681869, 0.9649115, 0.9681785, -0.0029752, 0.0030670
3: -0.0118319, 0.0116583, -0.0125017, 0.0115959, -0.0219447, 0.0226219
4: -0.0015797, 0.0002069, -0.0015750, 0.0002578, -0.0017205, 0.0016690
5: 0.0156738, 0.0174794, 0.0156786, 0.0175309, -0.0017389, 0.0016868
6: 0.0034048, 0.0042831, 0.0033798, 0.0042807, -0.0008205, 0.0008458
7: -0.0107996, -0.0047119, -0.0107834, -0.0045383, -0.0058627, 0.0056872
8: 0.0081613, 0.0129909, 0.0081741, 0.0131287, -0.0046512, 0.0045119
9: 0.0124034, 0.0210901, 0.0124265, 0.0213378, -0.0083656, 0.0081151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017972, upper bound: 0.0018708
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017972, upper bound: 0.0018708
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041854, -0.0041131, -0.0041906, -0.0041190, -0.0000665, 0.0000723
1: -0.0094000, -0.0066906, -0.0095948, -0.0069112, -0.0024888, 0.0027091
2: 0.9651830, 0.9684344, 0.9649493, 0.9681697, -0.0029867, 0.0032510
3: -0.0104978, 0.0134834, -0.0122222, 0.0115314, -0.0220292, 0.0239788
4: -0.0017185, 0.0001054, -0.0015701, 0.0002365, -0.0018237, 0.0016755
5: 0.0155335, 0.0173769, 0.0156835, 0.0175094, -0.0018432, 0.0016933
6: 0.0034547, 0.0043513, 0.0033902, 0.0042783, -0.0008236, 0.0008965
7: -0.0112726, -0.0050577, -0.0107667, -0.0046108, -0.0062143, 0.0057091
8: 0.0077860, 0.0127166, 0.0081873, 0.0130712, -0.0049302, 0.0045293
9: 0.0117285, 0.0205967, 0.0124503, 0.0212344, -0.0088674, 0.0081464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017989, upper bound: 0.0018708
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017989, upper bound: 0.0018708
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041895, -0.0041186, -0.0041909, -0.0041183, -0.0000653, 0.0000662
1: -0.0095507, -0.0068968, -0.0096039, -0.0068846, -0.0024464, 0.0024772
2: 0.9650022, 0.9681869, 0.9649384, 0.9682016, -0.0029358, 0.0029728
3: -0.0118319, 0.0116583, -0.0123026, 0.0117664, -0.0216541, 0.0219265
4: -0.0015797, 0.0002069, -0.0015879, 0.0002427, -0.0016676, 0.0016469
5: 0.0156738, 0.0174794, 0.0156655, 0.0175156, -0.0016854, 0.0016645
6: 0.0034048, 0.0042831, 0.0033872, 0.0042871, -0.0008096, 0.0008198
7: -0.0107996, -0.0047119, -0.0108276, -0.0045899, -0.0056824, 0.0056118
8: 0.0081613, 0.0129909, 0.0081390, 0.0130877, -0.0045082, 0.0044522
9: 0.0124034, 0.0210901, 0.0123634, 0.0212641, -0.0081084, 0.0080077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018316, upper bound: 0.0018996
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018316, upper bound: 0.0019019
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041854, -0.0041131, -0.0041899, -0.0041184, -0.0000670, 0.0000712
1: -0.0094000, -0.0066906, -0.0095683, -0.0068914, -0.0025086, 0.0026647
2: 0.9651830, 0.9684344, 0.9649811, 0.9681935, -0.0030104, 0.0031978
3: -0.0104978, 0.0134834, -0.0119873, 0.0117065, -0.0222043, 0.0235862
4: -0.0017185, 0.0001054, -0.0015834, 0.0002187, -0.0017939, 0.0016888
5: 0.0155335, 0.0173769, 0.0156701, 0.0174914, -0.0018130, 0.0017068
6: 0.0034547, 0.0043513, 0.0033990, 0.0042849, -0.0008302, 0.0008818
7: -0.0112726, -0.0050577, -0.0108121, -0.0046716, -0.0061126, 0.0057544
8: 0.0077860, 0.0127166, 0.0081513, 0.0130229, -0.0048494, 0.0045653
9: 0.0117285, 0.0205967, 0.0123856, 0.0211475, -0.0087221, 0.0082111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018376, upper bound: 0.0018996
time: 1.53 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018376, upper bound: 0.0019019
time: 1.03 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.88 seconds
IS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018494, upper bound: 0.0017625
IS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018494, upper bound: 0.0017625
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018622, upper bound: 0.0017680
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018622, upper bound: 0.0017680
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018539, upper bound: 0.0017779
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018539, upper bound: 0.0017779
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018539, upper bound: 0.0017779
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018539, upper bound: 0.0017779
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018910, upper bound: 0.0018087
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018910, upper bound: 0.0018097
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018920, upper bound: 0.0018087
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018920, upper bound: 0.0018097
IS_A2_B1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0018494
IS_A2_B1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0018494
IS_A2_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0017680, upper bound: 0.0018622
IS_A2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0017680, upper bound: 0.0018622
IS_A2_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0017680, upper bound: 0.0018622
IS_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0017680, upper bound: 0.0018622
IS_A2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0017779, upper bound: 0.0018539
IS_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0017779, upper bound: 0.0018539
IS_A2_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0017779, upper bound: 0.0018539
IS_A2_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0017779, upper bound: 0.0018539
IS_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018078, upper bound: 0.0018919
IS_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018078, upper bound: 0.0018920
IS_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018078, upper bound: 0.0018926
IS_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018078, upper bound: 0.0018927
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0017940, upper bound: 0.0018576
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0017940, upper bound: 0.0018576
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0017940, upper bound: 0.0018576
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0017940, upper bound: 0.0018576
IS_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018068, upper bound: 0.0018621
IS_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018068, upper bound: 0.0018621
IS_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018068, upper bound: 0.0018622
IS_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018068, upper bound: 0.0018622
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0017972, upper bound: 0.0018708
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0017972, upper bound: 0.0018708
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0017989, upper bound: 0.0018708
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0017989, upper bound: 0.0018708
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018316, upper bound: 0.0018996
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018316, upper bound: 0.0019019
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018376, upper bound: 0.0018996
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 2, lower bound: -0.0018376, upper bound: 0.0019019

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041895, -0.0041238, -0.0041895, -0.0041186, -0.0000710, 0.0000656
1: -0.0095541, -0.0070933, -0.0095507, -0.0068968, -0.0026572, 0.0024575
2: 0.9649981, 0.9679512, 0.9650022, 0.9681869, -0.0031888, 0.0029491
3: -0.0118616, 0.0099198, -0.0118319, 0.0116583, -0.0235199, 0.0217517
4: -0.0014475, 0.0002091, -0.0015797, 0.0002069, -0.0016543, 0.0017888
5: 0.0158074, 0.0174817, 0.0156738, 0.0174794, -0.0016720, 0.0018079
6: 0.0034037, 0.0042181, 0.0034048, 0.0042831, -0.0008794, 0.0008133
7: -0.0103491, -0.0047042, -0.0107996, -0.0047119, -0.0056372, 0.0060954
8: 0.0085187, 0.0129970, 0.0081613, 0.0129909, -0.0044723, 0.0048358
9: 0.0130463, 0.0211011, 0.0124034, 0.0210901, -0.0080438, 0.0086976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018165, upper bound: 0.0017651
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018495, upper bound: 0.0017780
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041895, -0.0041238, -0.0041854, -0.0041131, -0.0000755, 0.0000606
1: -0.0095541, -0.0070933, -0.0094000, -0.0066906, -0.0028273, 0.0022692
2: 0.9649981, 0.9679512, 0.9651830, 0.9684344, -0.0033929, 0.0027231
3: -0.0118616, 0.0099198, -0.0104978, 0.0134834, -0.0250253, 0.0200851
4: -0.0014475, 0.0002091, -0.0017185, 0.0001054, -0.0015276, 0.0019033
5: 0.0158074, 0.0174817, 0.0155335, 0.0173769, -0.0015439, 0.0019236
6: 0.0034037, 0.0042181, 0.0034547, 0.0043513, -0.0009357, 0.0007509
7: -0.0103491, -0.0047042, -0.0112726, -0.0050577, -0.0052052, 0.0064855
8: 0.0085187, 0.0129970, 0.0077860, 0.0127166, -0.0041296, 0.0051453
9: 0.0130463, 0.0211011, 0.0117285, 0.0205967, -0.0074274, 0.0092543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018391, upper bound: 0.0017455
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018495, upper bound: 0.0017780
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041889, -0.0041234, -0.0041899, -0.0041191, -0.0000699, 0.0000665
1: -0.0095312, -0.0070782, -0.0095685, -0.0069154, -0.0026158, 0.0024903
2: 0.9650257, 0.9679694, 0.9649808, 0.9681646, -0.0031390, 0.0029884
3: -0.0116592, 0.0100526, -0.0119893, 0.0114937, -0.0231529, 0.0220420
4: -0.0014576, 0.0001937, -0.0015672, 0.0002188, -0.0016764, 0.0017609
5: 0.0157972, 0.0174661, 0.0156864, 0.0174915, -0.0016943, 0.0017797
6: 0.0034113, 0.0042230, 0.0033989, 0.0042769, -0.0008656, 0.0008241
7: -0.0103835, -0.0047567, -0.0107569, -0.0046711, -0.0057124, 0.0060003
8: 0.0084914, 0.0129554, 0.0081951, 0.0130233, -0.0045319, 0.0047603
9: 0.0129972, 0.0210262, 0.0124643, 0.0211483, -0.0081511, 0.0085619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017922, upper bound: 0.0017507
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018281, upper bound: 0.0017625
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041889, -0.0041234, -0.0041864, -0.0041129, -0.0000759, 0.0000630
1: -0.0095312, -0.0070782, -0.0094383, -0.0066830, -0.0028425, 0.0023601
2: 0.9650257, 0.9679694, 0.9651371, 0.9684435, -0.0034111, 0.0028322
3: -0.0116592, 0.0100526, -0.0108370, 0.0135508, -0.0251596, 0.0208897
4: -0.0014576, 0.0001937, -0.0017236, 0.0001312, -0.0015888, 0.0019135
5: 0.0157972, 0.0174661, 0.0155283, 0.0174029, -0.0016057, 0.0019340
6: 0.0034113, 0.0042230, 0.0034420, 0.0043538, -0.0009407, 0.0007810
7: -0.0103835, -0.0047567, -0.0112900, -0.0049697, -0.0054137, 0.0065203
8: 0.0084914, 0.0129554, 0.0077722, 0.0127864, -0.0042950, 0.0051729
9: 0.0129972, 0.0210262, 0.0117036, 0.0207222, -0.0077250, 0.0093040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018172, upper bound: 0.0017277
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018281, upper bound: 0.0017625
time: 1.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041853, -0.0041170, -0.0041899, -0.0041191, -0.0000655, 0.0000722
1: -0.0093955, -0.0068383, -0.0095685, -0.0069154, -0.0024546, 0.0027053
2: 0.9651884, 0.9682573, 0.9649808, 0.9681646, -0.0029457, 0.0032465
3: -0.0104582, 0.0121768, -0.0119893, 0.0114937, -0.0217269, 0.0239459
4: -0.0016191, 0.0001024, -0.0015672, 0.0002188, -0.0018212, 0.0016525
5: 0.0156339, 0.0173738, 0.0156864, 0.0174915, -0.0018407, 0.0016701
6: 0.0034562, 0.0043025, 0.0033989, 0.0042769, -0.0008123, 0.0008953
7: -0.0109340, -0.0050679, -0.0107569, -0.0046711, -0.0062058, 0.0056307
8: 0.0080546, 0.0127085, 0.0081951, 0.0130233, -0.0049234, 0.0044671
9: 0.0122117, 0.0205821, 0.0124643, 0.0211483, -0.0088552, 0.0080346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017596, upper bound: 0.0017126
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018007, upper bound: 0.0017241
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041853, -0.0041170, -0.0041864, -0.0041129, -0.0000719, 0.0000693
1: -0.0093955, -0.0068383, -0.0094383, -0.0066830, -0.0026923, 0.0025969
2: 0.9651884, 0.9682573, 0.9651371, 0.9684435, -0.0032308, 0.0031164
3: -0.0104582, 0.0121768, -0.0108370, 0.0135508, -0.0238302, 0.0229862
4: -0.0016191, 0.0001024, -0.0017236, 0.0001312, -0.0017482, 0.0018124
5: 0.0156339, 0.0173738, 0.0155283, 0.0174029, -0.0017669, 0.0018318
6: 0.0034562, 0.0043025, 0.0034420, 0.0043538, -0.0008910, 0.0008594
7: -0.0109340, -0.0050679, -0.0112900, -0.0049697, -0.0059571, 0.0061758
8: 0.0080546, 0.0127085, 0.0077722, 0.0127864, -0.0047261, 0.0048996
9: 0.0122117, 0.0205821, 0.0117036, 0.0207222, -0.0085003, 0.0088124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017596, upper bound: 0.0017126
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018007, upper bound: 0.0017241
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041889, -0.0041234, -0.0041895, -0.0041186, -0.0000690, 0.0000648
1: -0.0095312, -0.0070782, -0.0095507, -0.0068968, -0.0025838, 0.0024271
2: 0.9650257, 0.9679694, 0.9650022, 0.9681869, -0.0031006, 0.0029126
3: -0.0116592, 0.0100526, -0.0118319, 0.0116583, -0.0228697, 0.0214832
4: -0.0014576, 0.0001937, -0.0015797, 0.0002069, -0.0016339, 0.0017394
5: 0.0157972, 0.0174661, 0.0156738, 0.0174794, -0.0016514, 0.0017579
6: 0.0034113, 0.0042230, 0.0034048, 0.0042831, -0.0008551, 0.0008032
7: -0.0103835, -0.0047567, -0.0107996, -0.0047119, -0.0055676, 0.0059269
8: 0.0084914, 0.0129554, 0.0081613, 0.0129909, -0.0044170, 0.0047021
9: 0.0129972, 0.0210262, 0.0124034, 0.0210901, -0.0079445, 0.0084572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018192, upper bound: 0.0017717
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018535, upper bound: 0.0017847
time: 1.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041889, -0.0041234, -0.0041854, -0.0041131, -0.0000744, 0.0000607
1: -0.0095312, -0.0070782, -0.0094000, -0.0066906, -0.0027855, 0.0022737
2: 0.9650257, 0.9679694, 0.9651830, 0.9684344, -0.0033427, 0.0027286
3: -0.0116592, 0.0100526, -0.0104978, 0.0134834, -0.0246551, 0.0201254
4: -0.0014576, 0.0001937, -0.0017185, 0.0001054, -0.0015307, 0.0018752
5: 0.0157972, 0.0174661, 0.0155335, 0.0173769, -0.0015470, 0.0018952
6: 0.0034113, 0.0042230, 0.0034547, 0.0043513, -0.0009218, 0.0007525
7: -0.0103835, -0.0047567, -0.0112726, -0.0050577, -0.0052157, 0.0063896
8: 0.0084914, 0.0129554, 0.0077860, 0.0127166, -0.0041379, 0.0050692
9: 0.0129972, 0.0210262, 0.0117285, 0.0205967, -0.0074423, 0.0091174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018418, upper bound: 0.0017515
time: 1.47 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018535, upper bound: 0.0017847
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041853, -0.0041170, -0.0041895, -0.0041186, -0.0000654, 0.0000711
1: -0.0093955, -0.0068383, -0.0095507, -0.0068968, -0.0024481, 0.0026640
2: 0.9651884, 0.9682573, 0.9650022, 0.9681869, -0.0029378, 0.0031969
3: -0.0104582, 0.0121768, -0.0118319, 0.0116583, -0.0216690, 0.0235798
4: -0.0016191, 0.0001024, -0.0015797, 0.0002069, -0.0017934, 0.0016481
5: 0.0156339, 0.0173738, 0.0156738, 0.0174794, -0.0018125, 0.0016656
6: 0.0034562, 0.0043025, 0.0034048, 0.0042831, -0.0008102, 0.0008816
7: -0.0109340, -0.0050679, -0.0107996, -0.0047119, -0.0061109, 0.0056157
8: 0.0080546, 0.0127085, 0.0081613, 0.0129909, -0.0048481, 0.0044552
9: 0.0122117, 0.0205821, 0.0124034, 0.0210901, -0.0087198, 0.0080132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017997, upper bound: 0.0017406
time: 1.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018414, upper bound: 0.0017560
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041853, -0.0041170, -0.0041854, -0.0041131, -0.0000701, 0.0000671
1: -0.0093955, -0.0068383, -0.0094000, -0.0066906, -0.0026259, 0.0025123
2: 0.9651884, 0.9682573, 0.9651830, 0.9684344, -0.0031511, 0.0030149
3: -0.0104582, 0.0121768, -0.0104978, 0.0134834, -0.0232423, 0.0222374
4: -0.0016191, 0.0001024, -0.0017185, 0.0001054, -0.0016913, 0.0017677
5: 0.0156339, 0.0173738, 0.0155335, 0.0173769, -0.0017093, 0.0017866
6: 0.0034562, 0.0043025, 0.0034547, 0.0043513, -0.0008690, 0.0008314
7: -0.0109340, -0.0050679, -0.0112726, -0.0050577, -0.0057630, 0.0060235
8: 0.0080546, 0.0127085, 0.0077860, 0.0127166, -0.0045721, 0.0047787
9: 0.0122117, 0.0205821, 0.0117285, 0.0205967, -0.0082234, 0.0085950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017997, upper bound: 0.0017406
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018414, upper bound: 0.0017560
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041895, -0.0041186, -0.0041895, -0.0041238, -0.0000656, 0.0000710
1: -0.0095507, -0.0068968, -0.0095541, -0.0070933, -0.0024575, 0.0026572
2: 0.9650022, 0.9681869, 0.9649981, 0.9679512, -0.0029491, 0.0031888
3: -0.0118319, 0.0116583, -0.0118616, 0.0099198, -0.0217517, 0.0235199
4: -0.0015797, 0.0002069, -0.0014475, 0.0002091, -0.0017888, 0.0016543
5: 0.0156738, 0.0174794, 0.0158074, 0.0174817, -0.0018079, 0.0016720
6: 0.0034048, 0.0042831, 0.0034037, 0.0042181, -0.0008133, 0.0008794
7: -0.0107996, -0.0047119, -0.0103491, -0.0047042, -0.0060954, 0.0056372
8: 0.0081613, 0.0129909, 0.0085187, 0.0129970, -0.0048358, 0.0044723
9: 0.0124034, 0.0210901, 0.0130463, 0.0211011, -0.0086976, 0.0080438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017651, upper bound: 0.0018165
time: 1.12 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017780, upper bound: 0.0018495
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041854, -0.0041131, -0.0041895, -0.0041238, -0.0000606, 0.0000755
1: -0.0094000, -0.0066906, -0.0095541, -0.0070933, -0.0022692, 0.0028273
2: 0.9651830, 0.9684344, 0.9649981, 0.9679512, -0.0027231, 0.0033929
3: -0.0104978, 0.0134834, -0.0118616, 0.0099198, -0.0200851, 0.0250253
4: -0.0017185, 0.0001054, -0.0014475, 0.0002091, -0.0019033, 0.0015276
5: 0.0155335, 0.0173769, 0.0158074, 0.0174817, -0.0019236, 0.0015439
6: 0.0034547, 0.0043513, 0.0034037, 0.0042181, -0.0007509, 0.0009357
7: -0.0112726, -0.0050577, -0.0103491, -0.0047042, -0.0064855, 0.0052052
8: 0.0077860, 0.0127166, 0.0085187, 0.0129970, -0.0051453, 0.0041296
9: 0.0117285, 0.0205967, 0.0130463, 0.0211011, -0.0092543, 0.0074274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017455, upper bound: 0.0018391
time: 1.10 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017780, upper bound: 0.0018495
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041895, -0.0041186, -0.0041866, -0.0041166, -0.0000726, 0.0000680
1: -0.0095507, -0.0068968, -0.0094426, -0.0068234, -0.0027203, 0.0025457
2: 0.9650022, 0.9681869, 0.9651320, 0.9682752, -0.0032645, 0.0030549
3: -0.0118319, 0.0116583, -0.0108748, 0.0123087, -0.0240783, 0.0225331
4: -0.0015797, 0.0002069, -0.0016292, 0.0001341, -0.0017138, 0.0018313
5: 0.0156738, 0.0174794, 0.0156238, 0.0174058, -0.0017321, 0.0018508
6: 0.0034048, 0.0042831, 0.0034406, 0.0043074, -0.0009002, 0.0008425
7: -0.0107996, -0.0047119, -0.0109682, -0.0049600, -0.0058396, 0.0062401
8: 0.0081613, 0.0129909, 0.0080275, 0.0127942, -0.0046329, 0.0049506
9: 0.0124034, 0.0210901, 0.0121629, 0.0207361, -0.0083327, 0.0089041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017003, upper bound: 0.0017670
time: 1.05 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017137, upper bound: 0.0018095
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041854, -0.0041131, -0.0041866, -0.0041166, -0.0000683, 0.0000729
1: -0.0094000, -0.0066906, -0.0094426, -0.0068234, -0.0025590, 0.0027307
2: 0.9651830, 0.9684344, 0.9651320, 0.9682752, -0.0030709, 0.0032770
3: -0.0104978, 0.0134834, -0.0108748, 0.0123087, -0.0226506, 0.0241703
4: -0.0017185, 0.0001054, -0.0016292, 0.0001341, -0.0018383, 0.0017227
5: 0.0155335, 0.0173769, 0.0156238, 0.0174058, -0.0018579, 0.0017411
6: 0.0034547, 0.0043513, 0.0034406, 0.0043074, -0.0008469, 0.0009037
7: -0.0112726, -0.0050577, -0.0109682, -0.0049600, -0.0062639, 0.0058701
8: 0.0077860, 0.0127166, 0.0080275, 0.0127942, -0.0049695, 0.0046571
9: 0.0117285, 0.0205967, 0.0121629, 0.0207361, -0.0089382, 0.0083762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016793, upper bound: 0.0017974
time: 1.04 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017137, upper bound: 0.0018095
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041899, -0.0041191, -0.0041889, -0.0041234, -0.0000665, 0.0000699
1: -0.0095685, -0.0069154, -0.0095312, -0.0070782, -0.0024903, 0.0026158
2: 0.9649808, 0.9681646, 0.9650257, 0.9679694, -0.0029884, 0.0031390
3: -0.0119893, 0.0114937, -0.0116592, 0.0100526, -0.0220420, 0.0231529
4: -0.0015672, 0.0002188, -0.0014576, 0.0001937, -0.0017609, 0.0016764
5: 0.0156864, 0.0174915, 0.0157972, 0.0174661, -0.0017797, 0.0016943
6: 0.0033989, 0.0042769, 0.0034113, 0.0042230, -0.0008241, 0.0008656
7: -0.0107569, -0.0046711, -0.0103835, -0.0047567, -0.0060003, 0.0057124
8: 0.0081951, 0.0130233, 0.0084914, 0.0129554, -0.0047603, 0.0045319
9: 0.0124643, 0.0211483, 0.0129972, 0.0210262, -0.0085619, 0.0081511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017507, upper bound: 0.0017922
time: 1.10 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0018281
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041864, -0.0041129, -0.0041889, -0.0041234, -0.0000630, 0.0000759
1: -0.0094383, -0.0066830, -0.0095312, -0.0070782, -0.0023601, 0.0028425
2: 0.9651371, 0.9684435, 0.9650257, 0.9679694, -0.0028322, 0.0034111
3: -0.0108370, 0.0135508, -0.0116592, 0.0100526, -0.0208897, 0.0251596
4: -0.0017236, 0.0001312, -0.0014576, 0.0001937, -0.0019135, 0.0015888
5: 0.0155283, 0.0174029, 0.0157972, 0.0174661, -0.0019340, 0.0016057
6: 0.0034420, 0.0043538, 0.0034113, 0.0042230, -0.0007810, 0.0009407
7: -0.0112900, -0.0049697, -0.0103835, -0.0047567, -0.0065203, 0.0054137
8: 0.0077722, 0.0127864, 0.0084914, 0.0129554, -0.0051729, 0.0042950
9: 0.0117036, 0.0207222, 0.0129972, 0.0210262, -0.0093040, 0.0077250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017277, upper bound: 0.0018172
time: 1.39 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0018281
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041899, -0.0041191, -0.0041853, -0.0041170, -0.0000722, 0.0000655
1: -0.0095685, -0.0069154, -0.0093955, -0.0068383, -0.0027053, 0.0024546
2: 0.9649808, 0.9681646, 0.9651884, 0.9682573, -0.0032465, 0.0029457
3: -0.0119893, 0.0114937, -0.0104582, 0.0121768, -0.0239459, 0.0217269
4: -0.0015672, 0.0002188, -0.0016191, 0.0001024, -0.0016525, 0.0018212
5: 0.0156864, 0.0174915, 0.0156339, 0.0173738, -0.0016701, 0.0018407
6: 0.0033989, 0.0042769, 0.0034562, 0.0043025, -0.0008953, 0.0008123
7: -0.0107569, -0.0046711, -0.0109340, -0.0050679, -0.0056307, 0.0062058
8: 0.0081951, 0.0130233, 0.0080546, 0.0127085, -0.0044671, 0.0049234
9: 0.0124643, 0.0211483, 0.0122117, 0.0205821, -0.0080346, 0.0088552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017126, upper bound: 0.0017596
time: 1.01 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017241, upper bound: 0.0018007
time: 1.38 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041864, -0.0041129, -0.0041853, -0.0041170, -0.0000693, 0.0000719
1: -0.0094383, -0.0066830, -0.0093955, -0.0068383, -0.0025969, 0.0026923
2: 0.9651371, 0.9684435, 0.9651884, 0.9682573, -0.0031164, 0.0032308
3: -0.0108370, 0.0135508, -0.0104582, 0.0121768, -0.0229862, 0.0238302
4: -0.0017236, 0.0001312, -0.0016191, 0.0001024, -0.0018124, 0.0017482
5: 0.0155283, 0.0174029, 0.0156339, 0.0173738, -0.0018318, 0.0017669
6: 0.0034420, 0.0043538, 0.0034562, 0.0043025, -0.0008594, 0.0008910
7: -0.0112900, -0.0049697, -0.0109340, -0.0050679, -0.0061758, 0.0059571
8: 0.0077722, 0.0127864, 0.0080546, 0.0127085, -0.0048996, 0.0047261
9: 0.0117036, 0.0207222, 0.0122117, 0.0205821, -0.0088124, 0.0085003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_B2_A1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017126, upper bound: 0.0017596
time: 1.44 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017241, upper bound: 0.0018007
time: 1.36 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041895, -0.0041186, -0.0041889, -0.0041234, -0.0000648, 0.0000690
1: -0.0095507, -0.0068968, -0.0095312, -0.0070782, -0.0024271, 0.0025838
2: 0.9650022, 0.9681869, 0.9650257, 0.9679694, -0.0029126, 0.0031006
3: -0.0118319, 0.0116583, -0.0116592, 0.0100526, -0.0214832, 0.0228697
4: -0.0015797, 0.0002069, -0.0014576, 0.0001937, -0.0017394, 0.0016339
5: 0.0156738, 0.0174794, 0.0157972, 0.0174661, -0.0017579, 0.0016514
6: 0.0034048, 0.0042831, 0.0034113, 0.0042230, -0.0008032, 0.0008551
7: -0.0107996, -0.0047119, -0.0103835, -0.0047567, -0.0059269, 0.0055676
8: 0.0081613, 0.0129909, 0.0084914, 0.0129554, -0.0047021, 0.0044170
9: 0.0124034, 0.0210901, 0.0129972, 0.0210262, -0.0084572, 0.0079445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017724, upper bound: 0.0018282
time: 1.10 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017859, upper bound: 0.0018610
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041854, -0.0041131, -0.0041889, -0.0041234, -0.0000607, 0.0000744
1: -0.0094000, -0.0066906, -0.0095312, -0.0070782, -0.0022737, 0.0027855
2: 0.9651830, 0.9684344, 0.9650257, 0.9679694, -0.0027286, 0.0033427
3: -0.0104978, 0.0134834, -0.0116592, 0.0100526, -0.0201254, 0.0246551
4: -0.0017185, 0.0001054, -0.0014576, 0.0001937, -0.0018752, 0.0015307
5: 0.0155335, 0.0173769, 0.0157972, 0.0174661, -0.0018952, 0.0015470
6: 0.0034547, 0.0043513, 0.0034113, 0.0042230, -0.0007525, 0.0009218
7: -0.0112726, -0.0050577, -0.0103835, -0.0047567, -0.0063896, 0.0052157
8: 0.0077860, 0.0127166, 0.0084914, 0.0129554, -0.0050692, 0.0041379
9: 0.0117285, 0.0205967, 0.0129972, 0.0210262, -0.0091174, 0.0074423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017534, upper bound: 0.0018505
time: 1.06 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017859, upper bound: 0.0018610
time: 1.53 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041895, -0.0041186, -0.0041853, -0.0041170, -0.0000711, 0.0000654
1: -0.0095507, -0.0068968, -0.0093955, -0.0068383, -0.0026640, 0.0024481
2: 0.9650022, 0.9681869, 0.9651884, 0.9682573, -0.0031969, 0.0029378
3: -0.0118319, 0.0116583, -0.0104582, 0.0121768, -0.0235798, 0.0216690
4: -0.0015797, 0.0002069, -0.0016191, 0.0001024, -0.0016481, 0.0017934
5: 0.0156738, 0.0174794, 0.0156339, 0.0173738, -0.0016656, 0.0018125
6: 0.0034048, 0.0042831, 0.0034562, 0.0043025, -0.0008816, 0.0008102
7: -0.0107996, -0.0047119, -0.0109340, -0.0050679, -0.0056157, 0.0061109
8: 0.0081613, 0.0129909, 0.0080546, 0.0127085, -0.0044552, 0.0048481
9: 0.0124034, 0.0210901, 0.0122117, 0.0205821, -0.0080132, 0.0087198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017388, upper bound: 0.0018003
time: 1.05 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017549, upper bound: 0.0018420
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041854, -0.0041131, -0.0041853, -0.0041170, -0.0000671, 0.0000701
1: -0.0094000, -0.0066906, -0.0093955, -0.0068383, -0.0025123, 0.0026259
2: 0.9651830, 0.9684344, 0.9651884, 0.9682573, -0.0030149, 0.0031511
3: -0.0104978, 0.0134834, -0.0104582, 0.0121768, -0.0222374, 0.0232423
4: -0.0017185, 0.0001054, -0.0016191, 0.0001024, -0.0017677, 0.0016913
5: 0.0155335, 0.0173769, 0.0156339, 0.0173738, -0.0017866, 0.0017093
6: 0.0034547, 0.0043513, 0.0034562, 0.0043025, -0.0008314, 0.0008690
7: -0.0112726, -0.0050577, -0.0109340, -0.0050679, -0.0060234, 0.0057630
8: 0.0077860, 0.0127166, 0.0080546, 0.0127085, -0.0047787, 0.0045721
9: 0.0117285, 0.0205967, 0.0122117, 0.0205821, -0.0085950, 0.0082234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017388, upper bound: 0.0018003
time: 1.56 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017549, upper bound: 0.0018416
time: 1.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041899, -0.0041191, -0.0041899, -0.0041191, -0.0000638, 0.0000638
1: -0.0095685, -0.0069154, -0.0095685, -0.0069154, -0.0023900, 0.0023900
2: 0.9649808, 0.9681646, 0.9649808, 0.9681646, -0.0028681, 0.0028681
3: -0.0119893, 0.0114937, -0.0119893, 0.0114937, -0.0211549, 0.0211549
4: -0.0015672, 0.0002188, -0.0015672, 0.0002188, -0.0016090, 0.0016090
5: 0.0156864, 0.0174915, 0.0156864, 0.0174915, -0.0016261, 0.0016261
6: 0.0033989, 0.0042769, 0.0033989, 0.0042769, -0.0007909, 0.0007909
7: -0.0107569, -0.0046711, -0.0107569, -0.0046711, -0.0054825, 0.0054825
8: 0.0081951, 0.0130233, 0.0081951, 0.0130233, -0.0043495, 0.0043495
9: 0.0124643, 0.0211483, 0.0124643, 0.0211483, -0.0078231, 0.0078231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017450, upper bound: 0.0018077
time: 1.39 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017612, upper bound: 0.0018526
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041899, -0.0041191, -0.0041864, -0.0041129, -0.0000703, 0.0000606
1: -0.0095685, -0.0069154, -0.0094383, -0.0066830, -0.0026329, 0.0022704
2: 0.9649808, 0.9681646, 0.9651371, 0.9684435, -0.0031596, 0.0027246
3: -0.0119893, 0.0114937, -0.0108370, 0.0135508, -0.0233048, 0.0200963
4: -0.0015672, 0.0002188, -0.0017236, 0.0001312, -0.0015284, 0.0017725
5: 0.0156864, 0.0174915, 0.0155283, 0.0174029, -0.0015448, 0.0017914
6: 0.0033989, 0.0042769, 0.0034420, 0.0043538, -0.0008713, 0.0007514
7: -0.0107569, -0.0046711, -0.0112900, -0.0049697, -0.0052081, 0.0060397
8: 0.0081951, 0.0130233, 0.0077722, 0.0127864, -0.0041319, 0.0047916
9: 0.0124643, 0.0211483, 0.0117036, 0.0207222, -0.0074316, 0.0086181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017450, upper bound: 0.0018078
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017612, upper bound: 0.0018526
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041864, -0.0041129, -0.0041899, -0.0041191, -0.0000606, 0.0000703
1: -0.0094383, -0.0066830, -0.0095685, -0.0069154, -0.0022704, 0.0026329
2: 0.9651371, 0.9684435, 0.9649808, 0.9681646, -0.0027246, 0.0031596
3: -0.0108370, 0.0135508, -0.0119893, 0.0114937, -0.0200963, 0.0233048
4: -0.0017236, 0.0001312, -0.0015672, 0.0002188, -0.0017725, 0.0015284
5: 0.0155283, 0.0174029, 0.0156864, 0.0174915, -0.0017914, 0.0015448
6: 0.0034420, 0.0043538, 0.0033989, 0.0042769, -0.0007514, 0.0008713
7: -0.0112900, -0.0049697, -0.0107569, -0.0046711, -0.0060397, 0.0052081
8: 0.0077722, 0.0127864, 0.0081951, 0.0130233, -0.0047916, 0.0041319
9: 0.0117036, 0.0207222, 0.0124643, 0.0211483, -0.0086181, 0.0074316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016934, upper bound: 0.0017883
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017421, upper bound: 0.0018058
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041864, -0.0041129, -0.0041864, -0.0041129, -0.0000657, 0.0000657
1: -0.0094383, -0.0066830, -0.0094383, -0.0066830, -0.0024604, 0.0024604
2: 0.9651371, 0.9684435, 0.9651371, 0.9684435, -0.0029526, 0.0029526
3: -0.0108370, 0.0135508, -0.0108370, 0.0135508, -0.0217779, 0.0217779
4: -0.0017236, 0.0001312, -0.0017236, 0.0001312, -0.0016563, 0.0016563
5: 0.0155283, 0.0174029, 0.0155283, 0.0174029, -0.0016740, 0.0016740
6: 0.0034420, 0.0043538, 0.0034420, 0.0043538, -0.0008142, 0.0008142
7: -0.0112900, -0.0049697, -0.0112900, -0.0049697, -0.0056439, 0.0056439
8: 0.0077722, 0.0127864, 0.0077722, 0.0127864, -0.0044776, 0.0044776
9: 0.0117036, 0.0207222, 0.0117036, 0.0207222, -0.0080534, 0.0080534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017214, upper bound: 0.0017557
time: 1.48 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017421, upper bound: 0.0018058
time: 1.40 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041899, -0.0041191, -0.0041895, -0.0041186, -0.0000668, 0.0000657
1: -0.0095685, -0.0069154, -0.0095507, -0.0068968, -0.0025023, 0.0024602
2: 0.9649808, 0.9681646, 0.9650022, 0.9681869, -0.0030029, 0.0029524
3: -0.0119893, 0.0114937, -0.0118319, 0.0116583, -0.0221486, 0.0217762
4: -0.0015672, 0.0002188, -0.0015797, 0.0002069, -0.0016562, 0.0016845
5: 0.0156864, 0.0174915, 0.0156738, 0.0174794, -0.0016739, 0.0017025
6: 0.0033989, 0.0042769, 0.0034048, 0.0042831, -0.0008281, 0.0008142
7: -0.0107569, -0.0046711, -0.0107996, -0.0047119, -0.0056435, 0.0057400
8: 0.0081951, 0.0130233, 0.0081613, 0.0129909, -0.0044773, 0.0045538
9: 0.0124643, 0.0211483, 0.0124034, 0.0210901, -0.0080528, 0.0081905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017644, upper bound: 0.0017942
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017815, upper bound: 0.0018338
time: 1.41 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041864, -0.0041129, -0.0041895, -0.0041186, -0.0000636, 0.0000722
1: -0.0094383, -0.0066830, -0.0095507, -0.0068968, -0.0023827, 0.0027031
2: 0.9651371, 0.9684435, 0.9650022, 0.9681869, -0.0028593, 0.0032439
3: -0.0108370, 0.0135508, -0.0118319, 0.0116583, -0.0210900, 0.0239262
4: -0.0017236, 0.0001312, -0.0015797, 0.0002069, -0.0018197, 0.0016040
5: 0.0155283, 0.0174029, 0.0156738, 0.0174794, -0.0018392, 0.0016211
6: 0.0034420, 0.0043538, 0.0034048, 0.0042831, -0.0007885, 0.0008946
7: -0.0112900, -0.0049697, -0.0107996, -0.0047119, -0.0062007, 0.0054657
8: 0.0077722, 0.0127864, 0.0081613, 0.0129909, -0.0049193, 0.0043362
9: 0.0117036, 0.0207222, 0.0124034, 0.0210901, -0.0088479, 0.0077991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017406, upper bound: 0.0018209
time: 1.34 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017815, upper bound: 0.0018338
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041899, -0.0041191, -0.0041854, -0.0041131, -0.0000718, 0.0000611
1: -0.0095685, -0.0069154, -0.0094000, -0.0066906, -0.0026884, 0.0022889
2: 0.9649808, 0.9681646, 0.9651830, 0.9684344, -0.0032262, 0.0027468
3: -0.0119893, 0.0114937, -0.0104978, 0.0134834, -0.0237960, 0.0202599
4: -0.0015672, 0.0002188, -0.0017185, 0.0001054, -0.0015409, 0.0018098
5: 0.0156864, 0.0174915, 0.0155335, 0.0173769, -0.0015573, 0.0018291
6: 0.0033989, 0.0042769, 0.0034547, 0.0043513, -0.0008897, 0.0007575
7: -0.0107569, -0.0046711, -0.0112726, -0.0050577, -0.0052505, 0.0061669
8: 0.0081951, 0.0130233, 0.0077860, 0.0127166, -0.0041655, 0.0048926
9: 0.0124643, 0.0211483, 0.0117285, 0.0205967, -0.0074921, 0.0087997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A1_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017354, upper bound: 0.0017633
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017551, upper bound: 0.0018107
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041864, -0.0041129, -0.0041854, -0.0041131, -0.0000687, 0.0000677
1: -0.0094383, -0.0066830, -0.0094000, -0.0066906, -0.0025723, 0.0025348
2: 0.9651371, 0.9684435, 0.9651830, 0.9684344, -0.0030869, 0.0030418
3: -0.0108370, 0.0135508, -0.0104978, 0.0134834, -0.0227686, 0.0224360
4: -0.0017236, 0.0001312, -0.0017185, 0.0001054, -0.0017064, 0.0017317
5: 0.0155283, 0.0174029, 0.0155335, 0.0173769, -0.0017246, 0.0017502
6: 0.0034420, 0.0043538, 0.0034547, 0.0043513, -0.0008513, 0.0008388
7: -0.0112900, -0.0049697, -0.0112726, -0.0050577, -0.0058145, 0.0059007
8: 0.0077722, 0.0127864, 0.0077860, 0.0127166, -0.0046129, 0.0046813
9: 0.0117036, 0.0207222, 0.0117285, 0.0205967, -0.0082968, 0.0084198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017354, upper bound: 0.0017633
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017551, upper bound: 0.0018106
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041895, -0.0041186, -0.0041899, -0.0041191, -0.0000657, 0.0000668
1: -0.0095507, -0.0068968, -0.0095685, -0.0069154, -0.0024602, 0.0025023
2: 0.9650022, 0.9681869, 0.9649808, 0.9681646, -0.0029524, 0.0030029
3: -0.0118319, 0.0116583, -0.0119893, 0.0114937, -0.0217762, 0.0221486
4: -0.0015797, 0.0002069, -0.0015672, 0.0002188, -0.0016845, 0.0016562
5: 0.0156738, 0.0174794, 0.0156864, 0.0174915, -0.0017025, 0.0016739
6: 0.0034048, 0.0042831, 0.0033989, 0.0042769, -0.0008142, 0.0008281
7: -0.0107996, -0.0047119, -0.0107569, -0.0046711, -0.0057400, 0.0056435
8: 0.0081613, 0.0129909, 0.0081951, 0.0130233, -0.0045538, 0.0044773
9: 0.0124034, 0.0210901, 0.0124643, 0.0211483, -0.0081905, 0.0080528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017220, upper bound: 0.0018258
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017611, upper bound: 0.0018413
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041895, -0.0041186, -0.0041864, -0.0041129, -0.0000722, 0.0000636
1: -0.0095507, -0.0068968, -0.0094383, -0.0066830, -0.0027031, 0.0023827
2: 0.9650022, 0.9681869, 0.9651371, 0.9684435, -0.0032439, 0.0028593
3: -0.0118319, 0.0116583, -0.0108370, 0.0135508, -0.0239262, 0.0210900
4: -0.0015797, 0.0002069, -0.0017236, 0.0001312, -0.0016040, 0.0018197
5: 0.0156738, 0.0174794, 0.0155283, 0.0174029, -0.0016211, 0.0018392
6: 0.0034048, 0.0042831, 0.0034420, 0.0043538, -0.0008946, 0.0007885
7: -0.0107996, -0.0047119, -0.0112900, -0.0049697, -0.0054657, 0.0062007
8: 0.0081613, 0.0129909, 0.0077722, 0.0127864, -0.0043362, 0.0049193
9: 0.0124034, 0.0210901, 0.0117036, 0.0207222, -0.0077991, 0.0088479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017450, upper bound: 0.0017947
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017612, upper bound: 0.0018413
time: 1.50 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041854, -0.0041131, -0.0041899, -0.0041191, -0.0000611, 0.0000718
1: -0.0094000, -0.0066906, -0.0095685, -0.0069154, -0.0022889, 0.0026884
2: 0.9651830, 0.9684344, 0.9649808, 0.9681646, -0.0027468, 0.0032262
3: -0.0104978, 0.0134834, -0.0119893, 0.0114937, -0.0202599, 0.0237960
4: -0.0017185, 0.0001054, -0.0015672, 0.0002188, -0.0018098, 0.0015409
5: 0.0155335, 0.0173769, 0.0156864, 0.0174915, -0.0018291, 0.0015573
6: 0.0034547, 0.0043513, 0.0033989, 0.0042769, -0.0007575, 0.0008897
7: -0.0112726, -0.0050577, -0.0107569, -0.0046711, -0.0061669, 0.0052505
8: 0.0077860, 0.0127166, 0.0081951, 0.0130233, -0.0048926, 0.0041655
9: 0.0117285, 0.0205967, 0.0124643, 0.0211483, -0.0087997, 0.0074921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017001, upper bound: 0.0018035
time: 1.39 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017468, upper bound: 0.0018200
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041854, -0.0041131, -0.0041864, -0.0041129, -0.0000677, 0.0000687
1: -0.0094000, -0.0066906, -0.0094383, -0.0066830, -0.0025348, 0.0025723
2: 0.9651830, 0.9684344, 0.9651371, 0.9684435, -0.0030418, 0.0030869
3: -0.0104978, 0.0134834, -0.0108370, 0.0135508, -0.0224360, 0.0227686
4: -0.0017185, 0.0001054, -0.0017236, 0.0001312, -0.0017317, 0.0017064
5: 0.0155335, 0.0173769, 0.0155283, 0.0174029, -0.0017502, 0.0017246
6: 0.0034547, 0.0043513, 0.0034420, 0.0043538, -0.0008388, 0.0008513
7: -0.0112726, -0.0050577, -0.0112900, -0.0049697, -0.0059007, 0.0058145
8: 0.0077860, 0.0127166, 0.0077722, 0.0127864, -0.0046813, 0.0046129
9: 0.0117285, 0.0205967, 0.0117036, 0.0207222, -0.0084198, 0.0082968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017001, upper bound: 0.0018035
time: 1.52 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017468, upper bound: 0.0018200
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041895, -0.0041186, -0.0041895, -0.0041186, -0.0000648, 0.0000648
1: -0.0095507, -0.0068968, -0.0095507, -0.0068968, -0.0024261, 0.0024261
2: 0.9650022, 0.9681869, 0.9650022, 0.9681869, -0.0029114, 0.0029114
3: -0.0118319, 0.0116583, -0.0118319, 0.0116583, -0.0214740, 0.0214740
4: -0.0015797, 0.0002069, -0.0015797, 0.0002069, -0.0016332, 0.0016332
5: 0.0156738, 0.0174794, 0.0156738, 0.0174794, -0.0016507, 0.0016507
6: 0.0034048, 0.0042831, 0.0034048, 0.0042831, -0.0008029, 0.0008029
7: -0.0107996, -0.0047119, -0.0107996, -0.0047119, -0.0055652, 0.0055652
8: 0.0081613, 0.0129909, 0.0081613, 0.0129909, -0.0044152, 0.0044152
9: 0.0124034, 0.0210901, 0.0124034, 0.0210901, -0.0079411, 0.0079411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017732, upper bound: 0.0018259
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017887, upper bound: 0.0018669
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041895, -0.0041186, -0.0041854, -0.0041131, -0.0000708, 0.0000613
1: -0.0095507, -0.0068968, -0.0094000, -0.0066906, -0.0026505, 0.0022969
2: 0.9650022, 0.9681869, 0.9651830, 0.9684344, -0.0031807, 0.0027564
3: -0.0118319, 0.0116583, -0.0104978, 0.0134834, -0.0234605, 0.0203305
4: -0.0015797, 0.0002069, -0.0017185, 0.0001054, -0.0015463, 0.0017843
5: 0.0156738, 0.0174794, 0.0155335, 0.0173769, -0.0015628, 0.0018034
6: 0.0034048, 0.0042831, 0.0034547, 0.0043513, -0.0008772, 0.0007601
7: -0.0107996, -0.0047119, -0.0112726, -0.0050577, -0.0052688, 0.0060800
8: 0.0081613, 0.0129909, 0.0077860, 0.0127166, -0.0041800, 0.0048236
9: 0.0124034, 0.0210901, 0.0117285, 0.0205967, -0.0075182, 0.0086757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017732, upper bound: 0.0018259
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017887, upper bound: 0.0018669
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041854, -0.0041131, -0.0041895, -0.0041186, -0.0000613, 0.0000708
1: -0.0094000, -0.0066906, -0.0095507, -0.0068968, -0.0022969, 0.0026505
2: 0.9651830, 0.9684344, 0.9650022, 0.9681869, -0.0027564, 0.0031807
3: -0.0104978, 0.0134834, -0.0118319, 0.0116583, -0.0203305, 0.0234605
4: -0.0017185, 0.0001054, -0.0015797, 0.0002069, -0.0017843, 0.0015463
5: 0.0155335, 0.0173769, 0.0156738, 0.0174794, -0.0018034, 0.0015628
6: 0.0034547, 0.0043513, 0.0034048, 0.0042831, -0.0007601, 0.0008772
7: -0.0112726, -0.0050577, -0.0107996, -0.0047119, -0.0060800, 0.0052688
8: 0.0077860, 0.0127166, 0.0081613, 0.0129909, -0.0048236, 0.0041800
9: 0.0117285, 0.0205967, 0.0124034, 0.0210901, -0.0086757, 0.0075182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017428, upper bound: 0.0018355
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017892, upper bound: 0.0018518
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041854, -0.0041131, -0.0041854, -0.0041131, -0.0000665, 0.0000665
1: -0.0094000, -0.0066906, -0.0094000, -0.0066906, -0.0024897, 0.0024897
2: 0.9651830, 0.9684344, 0.9651830, 0.9684344, -0.0029877, 0.0029877
3: -0.0104978, 0.0134834, -0.0104978, 0.0134834, -0.0220368, 0.0220368
4: -0.0017185, 0.0001054, -0.0017185, 0.0001054, -0.0016760, 0.0016760
5: 0.0155335, 0.0173769, 0.0155335, 0.0173769, -0.0016939, 0.0016939
6: 0.0034547, 0.0043513, 0.0034547, 0.0043513, -0.0008239, 0.0008239
7: -0.0112726, -0.0050577, -0.0112726, -0.0050577, -0.0057110, 0.0057110
8: 0.0077860, 0.0127166, 0.0077860, 0.0127166, -0.0045309, 0.0045309
9: 0.0117285, 0.0205967, 0.0117285, 0.0205967, -0.0081492, 0.0081492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017644, upper bound: 0.0018083
time: 1.46 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017892, upper bound: 0.0018518
time: 1.63 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.48 seconds
IS_A1_B2_A1_B2_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0018165, upper bound: 0.0017651
IS_A1_B2_A1_B2_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0018495, upper bound: 0.0017780
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0018391, upper bound: 0.0017455
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0018495, upper bound: 0.0017780
IS_A1_B2_A2_B1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017922, upper bound: 0.0017507
IS_A1_B2_A2_B1_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0018281, upper bound: 0.0017625
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0018172, upper bound: 0.0017277
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0018281, upper bound: 0.0017625
IS_A1_B2_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017596, upper bound: 0.0017126
IS_A1_B2_A2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0018007, upper bound: 0.0017241
IS_A1_B2_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017596, upper bound: 0.0017126
IS_A1_B2_A2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0018007, upper bound: 0.0017241
IS_A1_B2_A2_B2_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0018192, upper bound: 0.0017717
IS_A1_B2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0018535, upper bound: 0.0017847
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0018418, upper bound: 0.0017515
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0018535, upper bound: 0.0017847
IS_A1_B2_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017997, upper bound: 0.0017406
IS_A1_B2_A2_B2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0018414, upper bound: 0.0017560
IS_A1_B2_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017997, upper bound: 0.0017406
IS_A1_B2_A2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0018414, upper bound: 0.0017560
IS_A2_B1_B1_A2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017651, upper bound: 0.0018165
IS_A2_B1_B1_A2_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017780, upper bound: 0.0018495
IS_A2_B1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017455, upper bound: 0.0018391
IS_A2_B1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017780, upper bound: 0.0018495
IS_A2_B1_B1_A2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017003, upper bound: 0.0017670
IS_A2_B1_B1_A2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017137, upper bound: 0.0018095
IS_A2_B1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0016793, upper bound: 0.0017974
IS_A2_B1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017137, upper bound: 0.0018095
IS_A2_B1_B2_A1_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017507, upper bound: 0.0017922
IS_A2_B1_B2_A1_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0018281
IS_A2_B1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017277, upper bound: 0.0018172
IS_A2_B1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017625, upper bound: 0.0018281
IS_A2_B1_B2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017126, upper bound: 0.0017596
IS_A2_B1_B2_A1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017241, upper bound: 0.0018007
IS_A2_B1_B2_A1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017126, upper bound: 0.0017596
IS_A2_B1_B2_A1_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017241, upper bound: 0.0018007
IS_A2_B1_B2_A2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017724, upper bound: 0.0018282
IS_A2_B1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017859, upper bound: 0.0018610
IS_A2_B1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017534, upper bound: 0.0018505
IS_A2_B1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017859, upper bound: 0.0018610
IS_A2_B1_B2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017388, upper bound: 0.0018003
IS_A2_B1_B2_A2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017549, upper bound: 0.0018420
IS_A2_B1_B2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017388, upper bound: 0.0018003
IS_A2_B1_B2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017549, upper bound: 0.0018416
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017450, upper bound: 0.0018077
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017612, upper bound: 0.0018526
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017450, upper bound: 0.0018078
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017612, upper bound: 0.0018526
IS_A2_B2_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0016934, upper bound: 0.0017883
IS_A2_B2_A1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017421, upper bound: 0.0018058
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017214, upper bound: 0.0017557
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017421, upper bound: 0.0018058
IS_A2_B2_A1_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017644, upper bound: 0.0017942
IS_A2_B2_A1_B2_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017815, upper bound: 0.0018338
IS_A2_B2_A1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017406, upper bound: 0.0018209
IS_A2_B2_A1_B2_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017815, upper bound: 0.0018338
IS_A2_B2_A1_B2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017354, upper bound: 0.0017633
IS_A2_B2_A1_B2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017551, upper bound: 0.0018107
IS_A2_B2_A1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017354, upper bound: 0.0017633
IS_A2_B2_A1_B2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017551, upper bound: 0.0018106
IS_A2_B2_A2_B1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017220, upper bound: 0.0018258
IS_A2_B2_A2_B1_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017611, upper bound: 0.0018413
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017450, upper bound: 0.0017947
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017612, upper bound: 0.0018413
IS_A2_B2_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017001, upper bound: 0.0018035
IS_A2_B2_A2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017468, upper bound: 0.0018200
IS_A2_B2_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017001, upper bound: 0.0018035
IS_A2_B2_A2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017468, upper bound: 0.0018200
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017732, upper bound: 0.0018259
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017887, upper bound: 0.0018669
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017732, upper bound: 0.0018259
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017887, upper bound: 0.0018669
IS_A2_B2_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017428, upper bound: 0.0018355
IS_A2_B2_A2_B2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017892, upper bound: 0.0018518
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017644, upper bound: 0.0018083
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 2, lower bound: -0.0017892, upper bound: 0.0018518

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0041889, -0.0041234, -0.0041894, -0.0041199, -0.0000669, 0.0000648
1: -0.0095312, -0.0070782, -0.0095485, -0.0069451, -0.0025059, 0.0024252
2: 0.9650257, 0.9679694, 0.9650049, 0.9681291, -0.0030072, 0.0029104
3: -0.0116592, 0.0100526, -0.0118125, 0.0112315, -0.0221805, 0.0214664
4: -0.0014576, 0.0001937, -0.0015473, 0.0002054, -0.0016326, 0.0016870
5: 0.0157972, 0.0174661, 0.0157066, 0.0174779, -0.0016501, 0.0017050
6: 0.0034113, 0.0042230, 0.0034055, 0.0042671, -0.0008293, 0.0008026
7: -0.0103835, -0.0047567, -0.0106890, -0.0047169, -0.0055632, 0.0057483
8: 0.0084914, 0.0129554, 0.0082490, 0.0129870, -0.0044136, 0.0045604
9: 0.0129972, 0.0210262, 0.0125612, 0.0210829, -0.0079383, 0.0082023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018684, upper bound: 0.0018008
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018749, upper bound: 0.0018008
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041889, -0.0041246, -0.0041854, -0.0041131, -0.0000743, 0.0000588
1: -0.0095287, -0.0071236, -0.0094000, -0.0066906, -0.0027835, 0.0022017
2: 0.9650286, 0.9679148, 0.9651830, 0.9684344, -0.0033403, 0.0026421
3: -0.0116374, 0.0096511, -0.0104978, 0.0134834, -0.0246375, 0.0194879
4: -0.0014270, 0.0001921, -0.0017185, 0.0001054, -0.0014822, 0.0018738
5: 0.0158281, 0.0174645, 0.0155335, 0.0173769, -0.0014980, 0.0018938
6: 0.0034121, 0.0042080, 0.0034547, 0.0043513, -0.0009212, 0.0007286
7: -0.0102794, -0.0047623, -0.0112726, -0.0050577, -0.0050505, 0.0063850
8: 0.0085739, 0.0129509, 0.0077860, 0.0127166, -0.0040068, 0.0050656
9: 0.0131457, 0.0210181, 0.0117285, 0.0205967, -0.0072066, 0.0091109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017555, upper bound: 0.0016991
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017459, upper bound: 0.0016828
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0041894, -0.0041199, -0.0041889, -0.0041234, -0.0000648, 0.0000669
1: -0.0095485, -0.0069451, -0.0095312, -0.0070782, -0.0024252, 0.0025059
2: 0.9650049, 0.9681291, 0.9650257, 0.9679694, -0.0029104, 0.0030072
3: -0.0118125, 0.0112315, -0.0116592, 0.0100526, -0.0214664, 0.0221805
4: -0.0015473, 0.0002054, -0.0014576, 0.0001937, -0.0016870, 0.0016326
5: 0.0157066, 0.0174779, 0.0157972, 0.0174661, -0.0017050, 0.0016501
6: 0.0034055, 0.0042671, 0.0034113, 0.0042230, -0.0008026, 0.0008293
7: -0.0106890, -0.0047169, -0.0103835, -0.0047567, -0.0057483, 0.0055632
8: 0.0082490, 0.0129870, 0.0084914, 0.0129554, -0.0045604, 0.0044136
9: 0.0125612, 0.0210829, 0.0129972, 0.0210262, -0.0082023, 0.0079383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018001, upper bound: 0.0018700
time: 1.19 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018001, upper bound: 0.0018765
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041854, -0.0041131, -0.0041889, -0.0041246, -0.0000588, 0.0000743
1: -0.0094000, -0.0066906, -0.0095287, -0.0071236, -0.0022017, 0.0027835
2: 0.9651830, 0.9684344, 0.9650286, 0.9679148, -0.0026421, 0.0033403
3: -0.0104978, 0.0134834, -0.0116374, 0.0096511, -0.0194879, 0.0246375
4: -0.0017185, 0.0001054, -0.0014270, 0.0001921, -0.0018738, 0.0014822
5: 0.0155335, 0.0173769, 0.0158281, 0.0174645, -0.0018938, 0.0014980
6: 0.0034547, 0.0043513, 0.0034121, 0.0042080, -0.0007286, 0.0009212
7: -0.0112726, -0.0050577, -0.0102794, -0.0047623, -0.0063850, 0.0050505
8: 0.0077860, 0.0127166, 0.0085739, 0.0129509, -0.0050656, 0.0040068
9: 0.0117285, 0.0205967, 0.0131457, 0.0210181, -0.0091109, 0.0072066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017007, upper bound: 0.0017671
time: 1.11 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016848, upper bound: 0.0017574
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041894, -0.0041199, -0.0041895, -0.0041186, -0.0000647, 0.0000627
1: -0.0095485, -0.0069451, -0.0095507, -0.0068968, -0.0024241, 0.0023488
2: 0.9650049, 0.9681291, 0.9650022, 0.9681869, -0.0029090, 0.0028186
3: -0.0118125, 0.0112315, -0.0118319, 0.0116583, -0.0214564, 0.0207896
4: -0.0015473, 0.0002054, -0.0015797, 0.0002069, -0.0015812, 0.0016319
5: 0.0157066, 0.0174779, 0.0156738, 0.0174794, -0.0015981, 0.0016493
6: 0.0034055, 0.0042671, 0.0034048, 0.0042831, -0.0008022, 0.0007773
7: -0.0106890, -0.0047169, -0.0107996, -0.0047119, -0.0053878, 0.0055606
8: 0.0082490, 0.0129870, 0.0081613, 0.0129909, -0.0042744, 0.0044115
9: 0.0125612, 0.0210829, 0.0124034, 0.0210901, -0.0076880, 0.0079346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017989, upper bound: 0.0018768
time: 1.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018066, upper bound: 0.0018768
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041894, -0.0041199, -0.0041854, -0.0041131, -0.0000707, 0.0000592
1: -0.0095485, -0.0069451, -0.0094000, -0.0066906, -0.0026485, 0.0022151
2: 0.9650049, 0.9681291, 0.9651830, 0.9684344, -0.0031784, 0.0026582
3: -0.0118125, 0.0112315, -0.0104978, 0.0134834, -0.0234430, 0.0196063
4: -0.0015473, 0.0002054, -0.0017185, 0.0001054, -0.0014912, 0.0017830
5: 0.0157066, 0.0174779, 0.0155335, 0.0173769, -0.0015071, 0.0018020
6: 0.0034055, 0.0042671, 0.0034547, 0.0043513, -0.0008765, 0.0007330
7: -0.0106890, -0.0047169, -0.0112726, -0.0050577, -0.0050811, 0.0060754
8: 0.0082490, 0.0129870, 0.0077860, 0.0127166, -0.0040311, 0.0048200
9: 0.0125612, 0.0210829, 0.0117285, 0.0205967, -0.0072504, 0.0086692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016955, upper bound: 0.0017677
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016788, upper bound: 0.0017600
time: 1.38 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.81 seconds
IS_A1_B2_A2_B2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 2, lower bound: -0.0018684, upper bound: 0.0018008
IS_A1_B2_A2_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 2, lower bound: -0.0018749, upper bound: 0.0018008
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.81
Output dim: 2, lower bound: -0.0017555, upper bound: 0.0016991
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.81
Output dim: 2, lower bound: -0.0017459, upper bound: 0.0016828
IS_A2_B1_B2_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 2, lower bound: -0.0018001, upper bound: 0.0018700
IS_A2_B1_B2_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 2, lower bound: -0.0018001, upper bound: 0.0018765
IS_A2_B1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.81
Output dim: 2, lower bound: -0.0017007, upper bound: 0.0017671
IS_A2_B1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.81
Output dim: 2, lower bound: -0.0016848, upper bound: 0.0017574
IS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 2, lower bound: -0.0017989, upper bound: 0.0018768
IS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 2, lower bound: -0.0018066, upper bound: 0.0018768
IS_A2_B2_A2_B2_A1_B2_A2_A1, status: Status.VERIFIED, split count: 8, time: 3.81
Output dim: 2, lower bound: -0.0016955, upper bound: 0.0017677
IS_A2_B2_A2_B2_A1_B2_A2_A2, status: Status.VERIFIED, split count: 8, time: 3.81
Output dim: 2, lower bound: -0.0016788, upper bound: 0.0017600

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0041889, -0.0041234, -0.0041892, -0.0041219, -0.0000647, 0.0000646
1: -0.0095312, -0.0070782, -0.0095419, -0.0070199, -0.0024228, 0.0024204
2: 0.9650257, 0.9679694, 0.9650127, 0.9680393, -0.0029074, 0.0029046
3: -0.0116592, 0.0100526, -0.0117540, 0.0105695, -0.0214445, 0.0214241
4: -0.0014576, 0.0001937, -0.0014969, 0.0002009, -0.0016294, 0.0016310
5: 0.0157972, 0.0174661, 0.0157575, 0.0174734, -0.0016468, 0.0016484
6: 0.0034113, 0.0042230, 0.0034077, 0.0042424, -0.0008018, 0.0008010
7: -0.0103835, -0.0047567, -0.0105174, -0.0047321, -0.0055522, 0.0055575
8: 0.0084914, 0.0129554, 0.0083851, 0.0129749, -0.0044049, 0.0044091
9: 0.0129972, 0.0210262, 0.0128061, 0.0210613, -0.0079226, 0.0079302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018252, upper bound: 0.0017492
time: 1.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018252, upper bound: 0.0018007
time: 1.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0041889, -0.0041239, -0.0041914, -0.0041221, -0.0000620, 0.0000674
1: -0.0095299, -0.0070947, -0.0096229, -0.0070298, -0.0023225, 0.0025250
2: 0.9650272, 0.9679495, 0.9649156, 0.9680274, -0.0027871, 0.0030301
3: -0.0116477, 0.0099070, -0.0124708, 0.0104815, -0.0205568, 0.0223491
4: -0.0014465, 0.0001928, -0.0014902, 0.0002554, -0.0016998, 0.0015635
5: 0.0158084, 0.0174653, 0.0157642, 0.0175285, -0.0017179, 0.0015802
6: 0.0034117, 0.0042176, 0.0033809, 0.0042391, -0.0007686, 0.0008356
7: -0.0103457, -0.0047596, -0.0104946, -0.0045463, -0.0057920, 0.0053275
8: 0.0085213, 0.0129531, 0.0084032, 0.0131223, -0.0045951, 0.0042266
9: 0.0130510, 0.0210220, 0.0128386, 0.0213263, -0.0082647, 0.0076019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018219, upper bound: 0.0017471
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018219, upper bound: 0.0018008
time: 1.76 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0041892, -0.0041219, -0.0041889, -0.0041234, -0.0000646, 0.0000647
1: -0.0095419, -0.0070199, -0.0095312, -0.0070782, -0.0024204, 0.0024228
2: 0.9650127, 0.9680393, 0.9650257, 0.9679694, -0.0029046, 0.0029074
3: -0.0117540, 0.0105695, -0.0116592, 0.0100526, -0.0214241, 0.0214445
4: -0.0014969, 0.0002009, -0.0014576, 0.0001937, -0.0016310, 0.0016294
5: 0.0157575, 0.0174734, 0.0157972, 0.0174661, -0.0016484, 0.0016468
6: 0.0034077, 0.0042424, 0.0034113, 0.0042230, -0.0008010, 0.0008018
7: -0.0105174, -0.0047321, -0.0103835, -0.0047567, -0.0055575, 0.0055522
8: 0.0083851, 0.0129749, 0.0084914, 0.0129554, -0.0044091, 0.0044049
9: 0.0128061, 0.0210613, 0.0129972, 0.0210262, -0.0079302, 0.0079226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017490, upper bound: 0.0018485
time: 1.15 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017490, upper bound: 0.0018700
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0041914, -0.0041221, -0.0041889, -0.0041239, -0.0000674, 0.0000620
1: -0.0096229, -0.0070298, -0.0095299, -0.0070947, -0.0025250, 0.0023225
2: 0.9649156, 0.9680274, 0.9650272, 0.9679495, -0.0030301, 0.0027871
3: -0.0124708, 0.0104815, -0.0116477, 0.0099070, -0.0223491, 0.0205568
4: -0.0014902, 0.0002554, -0.0014465, 0.0001928, -0.0015635, 0.0016998
5: 0.0157642, 0.0175285, 0.0158084, 0.0174653, -0.0015802, 0.0017179
6: 0.0033809, 0.0042391, 0.0034117, 0.0042176, -0.0008356, 0.0007686
7: -0.0104946, -0.0045463, -0.0103457, -0.0047596, -0.0053275, 0.0057920
8: 0.0084032, 0.0131223, 0.0085213, 0.0129531, -0.0042266, 0.0045951
9: 0.0128386, 0.0213263, 0.0130510, 0.0210220, -0.0076019, 0.0082647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017467, upper bound: 0.0018554
time: 1.19 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017467, upper bound: 0.0018765
time: 1.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041894, -0.0041199, -0.0041893, -0.0041207, -0.0000623, 0.0000626
1: -0.0095485, -0.0069451, -0.0095443, -0.0069746, -0.0023340, 0.0023435
2: 0.9650049, 0.9681291, 0.9650099, 0.9680936, -0.0028009, 0.0028123
3: -0.0118125, 0.0112315, -0.0117748, 0.0109699, -0.0206591, 0.0207430
4: -0.0015473, 0.0002054, -0.0015274, 0.0002025, -0.0015776, 0.0015712
5: 0.0157066, 0.0174779, 0.0157267, 0.0174750, -0.0015945, 0.0015880
6: 0.0034055, 0.0042671, 0.0034069, 0.0042573, -0.0007724, 0.0007755
7: -0.0106890, -0.0047169, -0.0106212, -0.0047267, -0.0053757, 0.0053540
8: 0.0082490, 0.0129870, 0.0083028, 0.0129792, -0.0042648, 0.0042476
9: 0.0125612, 0.0210829, 0.0126580, 0.0210689, -0.0076707, 0.0076397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017570, upper bound: 0.0018562
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017570, upper bound: 0.0018768
time: 1.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041894, -0.0041204, -0.0041914, -0.0041210, -0.0000601, 0.0000655
1: -0.0095473, -0.0069661, -0.0096252, -0.0069888, -0.0022503, 0.0024526
2: 0.9650062, 0.9681039, 0.9649128, 0.9680766, -0.0027005, 0.0029432
3: -0.0118018, 0.0110455, -0.0124915, 0.0108443, -0.0199183, 0.0217087
4: -0.0015331, 0.0002046, -0.0015178, 0.0002570, -0.0016511, 0.0015149
5: 0.0157209, 0.0174771, 0.0157363, 0.0175301, -0.0016687, 0.0015311
6: 0.0034059, 0.0042602, 0.0033801, 0.0042526, -0.0007447, 0.0008117
7: -0.0106408, -0.0047197, -0.0105886, -0.0045410, -0.0056260, 0.0051620
8: 0.0082872, 0.0129848, 0.0083286, 0.0131265, -0.0044634, 0.0040953
9: 0.0126300, 0.0210789, 0.0127044, 0.0213340, -0.0080279, 0.0073658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017547, upper bound: 0.0018557
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017547, upper bound: 0.0018768
time: 1.83 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 4.36 seconds
IS_A1_B2_A2_B2_A1_B1_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.36
Output dim: 2, lower bound: -0.0018252, upper bound: 0.0017492
IS_A1_B2_A2_B2_A1_B1_B2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.36
Output dim: 2, lower bound: -0.0018252, upper bound: 0.0018007
IS_A1_B2_A2_B2_A1_B1_B2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.36
Output dim: 2, lower bound: -0.0018219, upper bound: 0.0017471
IS_A1_B2_A2_B2_A1_B1_B2_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.36
Output dim: 2, lower bound: -0.0018219, upper bound: 0.0018008
IS_A2_B1_B2_A2_B1_A1_A2_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.36
Output dim: 2, lower bound: -0.0017490, upper bound: 0.0018485
IS_A2_B1_B2_A2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 2, lower bound: -0.0017490, upper bound: 0.0018700
IS_A2_B1_B2_A2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 2, lower bound: -0.0017467, upper bound: 0.0018554
IS_A2_B1_B2_A2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 2, lower bound: -0.0017467, upper bound: 0.0018765
IS_A2_B2_A2_B2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 2, lower bound: -0.0017570, upper bound: 0.0018562
IS_A2_B2_A2_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 2, lower bound: -0.0017570, upper bound: 0.0018768
IS_A2_B2_A2_B2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 2, lower bound: -0.0017547, upper bound: 0.0018557
IS_A2_B2_A2_B2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 2, lower bound: -0.0017547, upper bound: 0.0018768

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041892, -0.0041219, -0.0041889, -0.0041246, -0.0000628, 0.0000646
1: -0.0095419, -0.0070199, -0.0095287, -0.0071236, -0.0023502, 0.0024207
2: 0.9650127, 0.9680393, 0.9650286, 0.9679148, -0.0028204, 0.0029049
3: -0.0117540, 0.0105695, -0.0116374, 0.0096511, -0.0208027, 0.0214264
4: -0.0014969, 0.0002009, -0.0014270, 0.0001921, -0.0016296, 0.0015822
5: 0.0157575, 0.0174734, 0.0158281, 0.0174645, -0.0016470, 0.0015991
6: 0.0034077, 0.0042424, 0.0034121, 0.0042080, -0.0007778, 0.0008011
7: -0.0105174, -0.0047321, -0.0102794, -0.0047623, -0.0055528, 0.0053912
8: 0.0083851, 0.0129749, 0.0085739, 0.0129509, -0.0044054, 0.0042771
9: 0.0128061, 0.0210613, 0.0131457, 0.0210181, -0.0079235, 0.0076928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017490, upper bound: 0.0018700
time: 1.15 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017490, upper bound: 0.0018700
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041914, -0.0041221, -0.0041894, -0.0041262, -0.0000648, 0.0000629
1: -0.0096229, -0.0070298, -0.0095479, -0.0071818, -0.0024248, 0.0023539
2: 0.9649156, 0.9680274, 0.9650056, 0.9678450, -0.0029099, 0.0028247
3: -0.0124708, 0.0104815, -0.0118066, 0.0091361, -0.0214630, 0.0208347
4: -0.0014902, 0.0002554, -0.0013879, 0.0002049, -0.0015846, 0.0016324
5: 0.0157642, 0.0175285, 0.0158676, 0.0174775, -0.0016015, 0.0016498
6: 0.0033809, 0.0042391, 0.0034058, 0.0041888, -0.0008025, 0.0007790
7: -0.0104946, -0.0045463, -0.0101459, -0.0047185, -0.0053995, 0.0055623
8: 0.0084032, 0.0131223, 0.0086798, 0.0129857, -0.0042837, 0.0044129
9: 0.0128386, 0.0213263, 0.0133361, 0.0210807, -0.0077047, 0.0079370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016527, upper bound: 0.0017820
time: 1.15 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016495, upper bound: 0.0017711
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041914, -0.0041221, -0.0041888, -0.0041251, -0.0000655, 0.0000620
1: -0.0096229, -0.0070298, -0.0095274, -0.0071397, -0.0024519, 0.0023204
2: 0.9649156, 0.9680274, 0.9650301, 0.9678954, -0.0029424, 0.0027845
3: -0.0124708, 0.0104815, -0.0116259, 0.0095083, -0.0217024, 0.0205383
4: -0.0014902, 0.0002554, -0.0014162, 0.0001912, -0.0015621, 0.0016506
5: 0.0157642, 0.0175285, 0.0158390, 0.0174636, -0.0015787, 0.0016682
6: 0.0033809, 0.0042391, 0.0034125, 0.0042027, -0.0008114, 0.0007679
7: -0.0104946, -0.0045463, -0.0102424, -0.0047653, -0.0053227, 0.0056244
8: 0.0084032, 0.0131223, 0.0086033, 0.0129486, -0.0042228, 0.0044621
9: 0.0128386, 0.0213263, 0.0131985, 0.0210139, -0.0075951, 0.0080255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016527, upper bound: 0.0018083
time: 1.60 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016495, upper bound: 0.0018016
time: 1.58 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0041894, -0.0041199, -0.0041897, -0.0041231, -0.0000572, 0.0000636
1: -0.0095485, -0.0069451, -0.0095592, -0.0070663, -0.0021408, 0.0023835
2: 0.9650049, 0.9681291, 0.9649920, 0.9679836, -0.0025691, 0.0028603
3: -0.0118125, 0.0112315, -0.0119069, 0.0101581, -0.0189490, 0.0210967
4: -0.0015473, 0.0002054, -0.0014656, 0.0002126, -0.0016045, 0.0014412
5: 0.0157066, 0.0174779, 0.0157891, 0.0174852, -0.0016217, 0.0014566
6: 0.0034055, 0.0042671, 0.0034020, 0.0042270, -0.0007085, 0.0007888
7: -0.0106890, -0.0047169, -0.0104108, -0.0046925, -0.0054674, 0.0049108
8: 0.0082490, 0.0129870, 0.0084697, 0.0130064, -0.0043376, 0.0038960
9: 0.0125612, 0.0210829, 0.0129582, 0.0211178, -0.0078015, 0.0070073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017570, upper bound: 0.0018543
time: 1.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017570, upper bound: 0.0018562
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0041894, -0.0041199, -0.0041892, -0.0041219, -0.0000604, 0.0000625
1: -0.0095485, -0.0069451, -0.0095419, -0.0070199, -0.0022616, 0.0023414
2: 0.9650049, 0.9681291, 0.9650127, 0.9680393, -0.0027141, 0.0028098
3: -0.0118125, 0.0112315, -0.0117540, 0.0105695, -0.0200184, 0.0207247
4: -0.0015473, 0.0002054, -0.0014969, 0.0002009, -0.0015762, 0.0015225
5: 0.0157066, 0.0174779, 0.0157575, 0.0174734, -0.0015931, 0.0015388
6: 0.0034055, 0.0042671, 0.0034077, 0.0042424, -0.0007485, 0.0007749
7: -0.0106890, -0.0047169, -0.0105174, -0.0047321, -0.0053710, 0.0051879
8: 0.0082490, 0.0129870, 0.0083851, 0.0129749, -0.0042611, 0.0041159
9: 0.0125612, 0.0210829, 0.0128061, 0.0210613, -0.0076640, 0.0074028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017570, upper bound: 0.0018727
time: 1.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017570, upper bound: 0.0018768
time: 1.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0041894, -0.0041204, -0.0041917, -0.0041235, -0.0000574, 0.0000665
1: -0.0095473, -0.0069661, -0.0096349, -0.0070808, -0.0021505, 0.0024904
2: 0.9650062, 0.9681039, 0.9649013, 0.9679663, -0.0025807, 0.0029886
3: -0.0118018, 0.0110455, -0.0125769, 0.0100304, -0.0190350, 0.0220430
4: -0.0015331, 0.0002046, -0.0014559, 0.0002635, -0.0016765, 0.0014477
5: 0.0157209, 0.0174771, 0.0157989, 0.0175367, -0.0016944, 0.0014632
6: 0.0034059, 0.0042602, 0.0033770, 0.0042222, -0.0007117, 0.0008242
7: -0.0106408, -0.0047197, -0.0103777, -0.0045188, -0.0057126, 0.0049331
8: 0.0082872, 0.0129848, 0.0084960, 0.0131441, -0.0045321, 0.0039137
9: 0.0126300, 0.0210789, 0.0130054, 0.0213656, -0.0081515, 0.0070391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016853, upper bound: 0.0017770
time: 1.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016645, upper bound: 0.0017714
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0041894, -0.0041204, -0.0041914, -0.0041221, -0.0000583, 0.0000654
1: -0.0095473, -0.0069661, -0.0096229, -0.0070298, -0.0021820, 0.0024505
2: 0.9650062, 0.9681039, 0.9649156, 0.9680274, -0.0026185, 0.0029407
3: -0.0118018, 0.0110455, -0.0124708, 0.0104815, -0.0193139, 0.0216903
4: -0.0015331, 0.0002046, -0.0014902, 0.0002554, -0.0016497, 0.0014689
5: 0.0157209, 0.0174771, 0.0157642, 0.0175285, -0.0016673, 0.0014846
6: 0.0034059, 0.0042602, 0.0033809, 0.0042391, -0.0007221, 0.0008110
7: -0.0106408, -0.0047197, -0.0104946, -0.0045463, -0.0056212, 0.0050054
8: 0.0082872, 0.0129848, 0.0084032, 0.0131223, -0.0044596, 0.0039710
9: 0.0126300, 0.0210789, 0.0128386, 0.0213263, -0.0080210, 0.0071423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016853, upper bound: 0.0018047
time: 1.60 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016645, upper bound: 0.0018025
time: 1.71 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 4.66 seconds
IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 10, time: 4.66
Output dim: 2, lower bound: -0.0017490, upper bound: 0.0018700
IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 10, time: 4.66
Output dim: 2, lower bound: -0.0017490, upper bound: 0.0018700
IS_A2_B1_B2_A2_B1_A1_A2_A2_B1_A1, status: Status.VERIFIED, split count: 10, time: 4.66
Output dim: 2, lower bound: -0.0016527, upper bound: 0.0017820
IS_A2_B1_B2_A2_B1_A1_A2_A2_B1_A2, status: Status.VERIFIED, split count: 10, time: 4.66
Output dim: 2, lower bound: -0.0016495, upper bound: 0.0017711
IS_A2_B1_B2_A2_B1_A1_A2_A2_B2_A1, status: Status.VERIFIED, split count: 10, time: 4.66
Output dim: 2, lower bound: -0.0016527, upper bound: 0.0018083
IS_A2_B1_B2_A2_B1_A1_A2_A2_B2_A2, status: Status.VERIFIED, split count: 10, time: 4.66
Output dim: 2, lower bound: -0.0016495, upper bound: 0.0018016
IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 10, time: 4.66
Output dim: 2, lower bound: -0.0017570, upper bound: 0.0018543
IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 4.66
Output dim: 2, lower bound: -0.0017570, upper bound: 0.0018562
IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 4.66
Output dim: 2, lower bound: -0.0017570, upper bound: 0.0018727
IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 4.66
Output dim: 2, lower bound: -0.0017570, upper bound: 0.0018768
IS_A2_B2_A2_B2_A1_B1_A2_B2_B1_B1, status: Status.VERIFIED, split count: 10, time: 4.66
Output dim: 2, lower bound: -0.0016853, upper bound: 0.0017770
IS_A2_B2_A2_B2_A1_B1_A2_B2_B1_B2, status: Status.VERIFIED, split count: 10, time: 4.66
Output dim: 2, lower bound: -0.0016645, upper bound: 0.0017714
IS_A2_B2_A2_B2_A1_B1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 10, time: 4.66
Output dim: 2, lower bound: -0.0016853, upper bound: 0.0018047
IS_A2_B2_A2_B2_A1_B1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 10, time: 4.66
Output dim: 2, lower bound: -0.0016645, upper bound: 0.0018025

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0041892, -0.0041219, -0.0041887, -0.0041263, -0.0000581, 0.0000645
1: -0.0095419, -0.0070199, -0.0095213, -0.0071840, -0.0021770, 0.0024153
2: 0.9650127, 0.9680393, 0.9650375, 0.9678424, -0.0026125, 0.0028984
3: -0.0117540, 0.0105695, -0.0115718, 0.0091163, -0.0192696, 0.0213784
4: -0.0014969, 0.0002009, -0.0013864, 0.0001871, -0.0016260, 0.0014656
5: 0.0157575, 0.0174734, 0.0158692, 0.0174594, -0.0016433, 0.0014812
6: 0.0034077, 0.0042424, 0.0034145, 0.0041880, -0.0007205, 0.0007993
7: -0.0105174, -0.0047321, -0.0101408, -0.0047793, -0.0055404, 0.0049939
8: 0.0083851, 0.0129749, 0.0086839, 0.0129375, -0.0043955, 0.0039619
9: 0.0128061, 0.0210613, 0.0133434, 0.0209939, -0.0079057, 0.0071259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017050, upper bound: 0.0018043
time: 1.11 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017063, upper bound: 0.0017995
time: 1.22 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0041892, -0.0041219, -0.0041909, -0.0041262, -0.0000589, 0.0000679
1: -0.0095419, -0.0070199, -0.0096045, -0.0071809, -0.0022051, 0.0025420
2: 0.9650127, 0.9680393, 0.9649377, 0.9678461, -0.0026462, 0.0030505
3: -0.0117540, 0.0105695, -0.0123080, 0.0091441, -0.0195178, 0.0225001
4: -0.0014969, 0.0002009, -0.0013885, 0.0002431, -0.0017113, 0.0014844
5: 0.0157575, 0.0174734, 0.0158670, 0.0175160, -0.0017295, 0.0015003
6: 0.0034077, 0.0042424, 0.0033870, 0.0041891, -0.0007297, 0.0008412
7: -0.0105174, -0.0047321, -0.0101480, -0.0045885, -0.0058311, 0.0050582
8: 0.0083851, 0.0129749, 0.0086782, 0.0130888, -0.0046261, 0.0040129
9: 0.0128061, 0.0210613, 0.0133332, 0.0212661, -0.0083205, 0.0072176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017178, upper bound: 0.0017998
time: 1.64 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017063, upper bound: 0.0017995
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041892, -0.0041219, -0.0041897, -0.0041231, -0.0000570, 0.0000614
1: -0.0095419, -0.0070199, -0.0095592, -0.0070663, -0.0021349, 0.0022982
2: 0.9650127, 0.9680393, 0.9649920, 0.9679836, -0.0025619, 0.0027580
3: -0.0117540, 0.0105695, -0.0119069, 0.0101581, -0.0188964, 0.0203424
4: -0.0014969, 0.0002009, -0.0014656, 0.0002126, -0.0015472, 0.0014372
5: 0.0157575, 0.0174734, 0.0157891, 0.0174852, -0.0015637, 0.0014525
6: 0.0034077, 0.0042424, 0.0034020, 0.0042270, -0.0007065, 0.0007606
7: -0.0105174, -0.0047321, -0.0104108, -0.0046925, -0.0052719, 0.0048972
8: 0.0083851, 0.0129749, 0.0084697, 0.0130064, -0.0041825, 0.0038852
9: 0.0128061, 0.0210613, 0.0129582, 0.0211178, -0.0075226, 0.0069879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016925, upper bound: 0.0017855
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016748, upper bound: 0.0017830
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041914, -0.0041221, -0.0041897, -0.0041231, -0.0000605, 0.0000593
1: -0.0096229, -0.0070298, -0.0095592, -0.0070663, -0.0022639, 0.0022214
2: 0.9649156, 0.9680274, 0.9649920, 0.9679836, -0.0027168, 0.0026658
3: -0.0124708, 0.0104815, -0.0119069, 0.0101581, -0.0200386, 0.0196626
4: -0.0014902, 0.0002554, -0.0014656, 0.0002126, -0.0014955, 0.0015241
5: 0.0157642, 0.0175285, 0.0157891, 0.0174852, -0.0015114, 0.0015403
6: 0.0033809, 0.0042391, 0.0034020, 0.0042270, -0.0007492, 0.0007352
7: -0.0104946, -0.0045463, -0.0104108, -0.0046925, -0.0050957, 0.0051932
8: 0.0084032, 0.0131223, 0.0084697, 0.0130064, -0.0040427, 0.0041200
9: 0.0128386, 0.0213263, 0.0129582, 0.0211178, -0.0072712, 0.0074103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016745, upper bound: 0.0017885
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016748, upper bound: 0.0017830
time: 1.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041892, -0.0041219, -0.0041892, -0.0041219, -0.0000603, 0.0000603
1: -0.0095419, -0.0070199, -0.0095419, -0.0070199, -0.0022563, 0.0022563
2: 0.9650127, 0.9680393, 0.9650127, 0.9680393, -0.0027077, 0.0027077
3: -0.0117540, 0.0105695, -0.0117540, 0.0105695, -0.0199714, 0.0199714
4: -0.0014969, 0.0002009, -0.0014969, 0.0002009, -0.0015189, 0.0015189
5: 0.0157575, 0.0174734, 0.0157575, 0.0174734, -0.0015352, 0.0015352
6: 0.0034077, 0.0042424, 0.0034077, 0.0042424, -0.0007467, 0.0007467
7: -0.0105174, -0.0047321, -0.0105174, -0.0047321, -0.0051758, 0.0051758
8: 0.0083851, 0.0129749, 0.0083851, 0.0129749, -0.0041062, 0.0041062
9: 0.0128061, 0.0210613, 0.0128061, 0.0210613, -0.0073854, 0.0073854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017164, upper bound: 0.0018075
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017190, upper bound: 0.0018028
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041914, -0.0041221, -0.0041892, -0.0041219, -0.0000636, 0.0000583
1: -0.0096229, -0.0070298, -0.0095419, -0.0070199, -0.0023816, 0.0021830
2: 0.9649156, 0.9680274, 0.9650127, 0.9680393, -0.0028580, 0.0026196
3: -0.0124708, 0.0104815, -0.0117540, 0.0105695, -0.0210800, 0.0193220
4: -0.0014902, 0.0002554, -0.0014969, 0.0002009, -0.0014696, 0.0016033
5: 0.0157642, 0.0175285, 0.0157575, 0.0174734, -0.0014852, 0.0016204
6: 0.0033809, 0.0042391, 0.0034077, 0.0042424, -0.0007881, 0.0007224
7: -0.0104946, -0.0045463, -0.0105174, -0.0047321, -0.0050075, 0.0054631
8: 0.0084032, 0.0131223, 0.0083851, 0.0129749, -0.0039727, 0.0043341
9: 0.0128386, 0.0213263, 0.0128061, 0.0210613, -0.0071453, 0.0077954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017164, upper bound: 0.0018095
time: 1.22 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017190, upper bound: 0.0018034
time: 1.27 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 3.88 seconds
IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.88
Output dim: 2, lower bound: -0.0017050, upper bound: 0.0018043
IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 11, time: 3.88
Output dim: 2, lower bound: -0.0017063, upper bound: 0.0017995
IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 11, time: 3.88
Output dim: 2, lower bound: -0.0017178, upper bound: 0.0017998
IS_A2_B1_B2_A2_B1_A1_A2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 11, time: 3.88
Output dim: 2, lower bound: -0.0017063, upper bound: 0.0017995
IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A1_B1, status: Status.VERIFIED, split count: 11, time: 3.88
Output dim: 2, lower bound: -0.0016925, upper bound: 0.0017855
IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 11, time: 3.88
Output dim: 2, lower bound: -0.0016748, upper bound: 0.0017830
IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 11, time: 3.88
Output dim: 2, lower bound: -0.0016745, upper bound: 0.0017885
IS_A2_B2_A2_B2_A1_B1_A2_B1_B1_A2_A2, status: Status.VERIFIED, split count: 11, time: 3.88
Output dim: 2, lower bound: -0.0016748, upper bound: 0.0017830
IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 11, time: 3.88
Output dim: 2, lower bound: -0.0017164, upper bound: 0.0018075
IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 11, time: 3.88
Output dim: 2, lower bound: -0.0017190, upper bound: 0.0018028
IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 11, time: 3.88
Output dim: 2, lower bound: -0.0017164, upper bound: 0.0018095
IS_A2_B2_A2_B2_A1_B1_A2_B1_B2_A2_A2, status: Status.VERIFIED, split count: 11, time: 3.88
Output dim: 2, lower bound: -0.0017190, upper bound: 0.0018034

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.31 + 437.74 = 441.05 seconds
