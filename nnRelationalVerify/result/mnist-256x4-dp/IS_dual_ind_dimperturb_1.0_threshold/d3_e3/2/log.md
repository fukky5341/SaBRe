## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 27.4911861951


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255)
1: (-20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821)
2: (-25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841)
3: (-29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502)
4: (-27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682)
5: (-20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750)
6: (-22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550)
7: (-26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649)
8: (-32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634)
9: (-19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.42 + 15.63 = 17.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -27.5187049, upper bound: 27.5187042

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 197

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5186564, upper bound: 27.5187018
time: 8.20 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5187049, upper bound: 27.5187046
time: 6.03 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.37 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 14.37
Output dim: 1, lower bound: -27.5186564, upper bound: 27.5187018
IS_A2, status: Status.UNKNOWN, split count: 1, time: 14.37
Output dim: 1, lower bound: -27.5187049, upper bound: 27.5187046

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -20.3372707, 18.3306427, -21.1561298, 19.0621166, -39.3993874, 39.4867706
1: -19.8265934, 12.6207695, -20.5984573, 13.1112080, -32.9378014, 33.2192268
2: -23.8497105, 15.6505747, -24.8135319, 16.2665253, -40.1162338, 40.4640999
3: -28.2927589, 13.6621714, -29.4291115, 14.1977253, -42.4904861, 43.0912819
4: -25.7933159, 16.6381626, -26.8228664, 17.3019867, -43.0952988, 43.4610252
5: -19.7755775, 17.7976685, -20.5701656, 18.5073338, -38.2828979, 38.3678284
6: -20.9945984, 19.1023159, -21.8345146, 19.8566895, -40.8512878, 40.9368286
7: -25.1642418, 18.4076538, -26.1630421, 19.1327705, -44.2970123, 44.5706940
8: -30.4107609, 15.4058342, -31.6357841, 16.0281162, -46.4388771, 47.0416183
9: -18.3709221, 20.6629963, -19.1117859, 21.4812775, -39.8521996, 39.7747803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=56, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=122, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=44, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5172199, upper bound: 27.5179418
time: 10.70 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5172073, upper bound: 27.5172466
time: 6.37 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -20.8341827, 18.7764797, -21.3716774, 19.2532825, -40.0874634, 40.1481552
1: -20.2967739, 12.9137506, -20.8013935, 13.2438803, -33.5406532, 33.7151413
2: -24.4328651, 16.0246868, -25.0681591, 16.4283524, -40.8612099, 41.0928421
3: -28.9881554, 13.9880733, -29.7231674, 14.3389206, -43.3270760, 43.7112427
4: -26.4227924, 17.0404854, -27.0904026, 17.4780807, -43.9008713, 44.1308823
5: -20.2581120, 18.2318153, -20.7789726, 18.6921959, -38.9503059, 39.0107880
6: -21.5058117, 19.5596523, -22.0539761, 20.0556049, -41.5614128, 41.6136284
7: -25.7741699, 18.8476753, -26.4229431, 19.3236465, -45.0978088, 45.2706108
8: -31.1600342, 15.7805538, -31.9533882, 16.1953068, -47.3553391, 47.7339401
9: -18.8187046, 21.1616440, -19.3084393, 21.6951752, -40.5138779, 40.4700851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=56, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5172612, upper bound: 27.5179477
time: 6.27 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5172485, upper bound: 27.5172484
time: 10.05 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 17.73 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 17.73
Output dim: 1, lower bound: -27.5172199, upper bound: 27.5179418
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 17.73
Output dim: 1, lower bound: -27.5172073, upper bound: 27.5172466
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 17.73
Output dim: 1, lower bound: -27.5172612, upper bound: 27.5179477
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 17.73
Output dim: 1, lower bound: -27.5172485, upper bound: 27.5172484

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -20.1438332, 18.1623688, -19.9846420, 18.0450287, -38.1888618, 38.1470108
1: -19.6454964, 12.4926529, -19.5033092, 12.3417282, -31.9872246, 31.9959602
2: -23.6165276, 15.5055523, -23.4053249, 15.3892546, -39.0057831, 38.9108772
3: -28.0367851, 13.5362148, -27.8762913, 13.4348240, -41.4716110, 41.4125061
4: -25.5595436, 16.4785309, -25.4029522, 16.3373661, -41.8969116, 41.8814850
5: -19.5891953, 17.6377716, -19.4427261, 17.5361137, -37.1253090, 37.0804977
6: -20.8019238, 18.9213390, -20.6664734, 18.7617188, -39.5636444, 39.5878105
7: -24.9356537, 18.2356339, -24.7741299, 18.0932121, -43.0288658, 43.0097656
8: -30.1383495, 15.2531528, -29.9830513, 15.1074095, -45.2457542, 45.2362022
9: -18.1936989, 20.4725990, -18.0406303, 20.3277531, -38.5214539, 38.5132294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=55, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=122, inp2_unstable=120, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=41, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5160050, upper bound: 27.5175559
time: 34.25 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5169536, upper bound: 27.5177486
time: 10.92 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -19.9862595, 18.0284519, -21.4197960, 19.3245296, -39.3107910, 39.4482498
1: -19.4985828, 12.3856688, -20.8300648, 13.1390085, -32.6375923, 33.2157288
2: -23.4301071, 15.3884487, -25.0801182, 16.4393730, -39.8694725, 40.4685669
3: -27.8360996, 13.4346247, -29.9399071, 14.3563185, -42.1924171, 43.3745308
4: -25.3745880, 16.3487091, -27.2429485, 17.4642658, -42.8388481, 43.5916595
5: -19.4402504, 17.5097198, -20.8299770, 18.7829628, -38.2232018, 38.3396912
6: -20.6486282, 18.7750969, -22.1528587, 20.0820827, -40.7307129, 40.9279556
7: -24.7564373, 18.0985126, -26.5666733, 19.3563995, -44.1128387, 44.6651802
8: -29.9222202, 15.1272440, -32.1334763, 16.1172371, -46.0394516, 47.2607155
9: -18.0509872, 20.3213749, -19.3229923, 21.7710381, -39.8220177, 39.6443672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=55, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=122, inp2_unstable=121, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 197

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5159920, upper bound: 27.5168373
time: 10.12 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5169407, upper bound: 27.5170007
time: 5.97 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -20.6329498, 18.6007214, -20.1923828, 18.2282448, -38.8611908, 38.7931061
1: -20.1084270, 12.7802858, -19.6989422, 12.4688091, -32.5772362, 32.4792252
2: -24.1897068, 15.8740463, -23.6493721, 15.5454712, -39.7351685, 39.5234184
3: -28.7215500, 13.8566017, -28.1598930, 13.5704517, -42.2919998, 42.0164948
4: -26.1784115, 16.8740406, -25.6597061, 16.5064564, -42.6848679, 42.5337448
5: -20.0627403, 18.0656700, -19.6424084, 17.7145481, -37.7772903, 37.7080688
6: -21.3048534, 19.3711491, -20.8773708, 18.9530735, -40.2579269, 40.2485161
7: -25.5359859, 18.6680889, -25.0245209, 18.2764854, -43.8124657, 43.6926079
8: -30.8759842, 15.6218929, -30.2887287, 15.2683735, -46.1443558, 45.9106216
9: -18.6330700, 20.9631195, -18.2284374, 20.5334625, -39.1665344, 39.1915588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=55, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=120, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=44, inp2_unstable=41, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 197

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5161434, upper bound: 27.5176149
time: 13.74 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5170179, upper bound: 27.5177586
time: 11.35 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -20.4647141, 18.4567566, -21.6273041, 19.5082817, -39.9729958, 40.0840569
1: -19.9512672, 12.6670856, -21.0257893, 13.2665472, -33.2178040, 33.6928749
2: -23.9904385, 15.7493877, -25.3249035, 16.5950985, -40.5855331, 41.0742836
3: -28.5056572, 13.7480879, -30.2231731, 14.4921398, -42.9977951, 43.9712601
4: -25.9795074, 16.7356758, -27.5004749, 17.6333714, -43.6128731, 44.2361526
5: -19.9031353, 17.9282894, -21.0301323, 18.9614449, -38.8645782, 38.9584198
6: -21.1404533, 19.2151222, -22.3638725, 20.2735004, -41.4139557, 41.5789948
7: -25.3436432, 18.5216103, -26.8171921, 19.5395184, -44.8831596, 45.3388023
8: -30.6435375, 15.4883432, -32.4389839, 16.2782478, -46.9217796, 47.9273262
9: -18.4803677, 20.8010178, -19.5114174, 21.9769211, -40.4572906, 40.3124352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=55, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=122, inp2_unstable=121, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5161303, upper bound: 27.5168617
time: 44.48 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5170040, upper bound: 27.5170040
time: 5.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 51.66 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 51.66
Output dim: 1, lower bound: -27.5160050, upper bound: 27.5175559
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 51.66
Output dim: 1, lower bound: -27.5169536, upper bound: 27.5177486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 51.66
Output dim: 1, lower bound: -27.5159920, upper bound: 27.5168373
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 51.66
Output dim: 1, lower bound: -27.5169407, upper bound: 27.5170007
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 51.66
Output dim: 1, lower bound: -27.5161434, upper bound: 27.5176149
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 51.66
Output dim: 1, lower bound: -27.5170179, upper bound: 27.5177586
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 51.66
Output dim: 1, lower bound: -27.5161303, upper bound: 27.5168617
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 51.66
Output dim: 1, lower bound: -27.5170040, upper bound: 27.5170040

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -18.2013054, 16.4866562, -19.3997593, 17.5405846, -35.7418861, 35.8864136
1: -17.8407879, 11.2107658, -18.9597225, 11.9570885, -29.7978725, 30.1704865
2: -21.3263588, 14.0511751, -22.7171745, 14.9520359, -36.2783966, 36.7683487
3: -25.4728603, 12.2558260, -27.1056080, 13.0488091, -38.5216675, 39.3614349
4: -23.2468185, 14.8770647, -24.7083721, 15.8567448, -39.1035614, 39.5854378
5: -17.7513828, 16.0112190, -18.8887444, 17.0477333, -34.7991180, 34.8999634
6: -18.8941250, 17.1186275, -20.0941830, 18.2184830, -37.1126099, 37.2128067
7: -22.6991920, 16.5269737, -24.1015205, 17.5801640, -40.2793579, 40.6284943
8: -27.4064026, 13.6893997, -29.1609268, 14.6387424, -42.0451431, 42.8503265
9: -16.4214363, 18.5750656, -17.5091629, 19.7576981, -36.1791229, 36.0842209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=115, inp2_unstable=119, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=39, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5160050, upper bound: 27.5174886
time: 5.13 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5160050, upper bound: 27.5175559
time: 6.03 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -19.0918045, 17.2597313, -19.7846737, 17.8732319, -36.9650345, 37.0444031
1: -18.6689243, 11.7919111, -19.3170757, 12.2087164, -30.8776398, 31.1089859
2: -22.3758392, 14.7180252, -23.1700115, 15.2394838, -37.6153145, 37.8880310
3: -26.6633892, 12.8471889, -27.6143188, 13.3034639, -39.9668427, 40.4615097
4: -24.3159409, 15.6121664, -25.1664963, 16.1728249, -40.4887543, 40.7786598
5: -18.5963440, 16.7650948, -19.2537899, 17.3698959, -35.9662399, 36.0188828
6: -19.7736626, 17.9449997, -20.4710007, 18.5761070, -38.3497696, 38.4160004
7: -23.7339211, 17.3144455, -24.5452881, 17.9180946, -41.6520119, 41.8597260
8: -28.6704998, 14.4042587, -29.7037239, 14.9462128, -43.6167145, 44.1079826
9: -17.2351589, 19.4499187, -17.8587551, 20.1329956, -37.3681526, 37.3086739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=119, inp2_unstable=120, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5169536, upper bound: 27.5176896
time: 11.09 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5169536, upper bound: 27.5177485
time: 16.11 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -18.0628338, 16.3677444, -20.8142815, 18.8034840, -36.8663177, 37.1820259
1: -17.7095165, 11.1171503, -20.2665615, 12.7423267, -30.4518433, 31.3837128
2: -21.1626053, 13.9469995, -24.3670387, 15.9890194, -37.1516266, 38.3140373
3: -25.2909832, 12.1655216, -29.1409492, 13.9592342, -39.2502136, 41.3064728
4: -23.0820427, 14.7623730, -26.5229416, 16.9691906, -40.0512314, 41.2853165
5: -17.6192284, 15.8962221, -20.2579536, 18.2785091, -35.8977356, 36.1541748
6: -18.7561798, 16.9904861, -21.5609798, 19.5212765, -38.2774544, 38.5514679
7: -22.5376930, 16.4048424, -25.8705120, 18.8267136, -41.3644028, 42.2753525
8: -27.2134533, 13.5776005, -31.2830143, 15.6355152, -42.8489609, 44.8606071
9: -16.2953796, 18.4381485, -18.7740059, 21.1812820, -37.4766617, 37.2121506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=115, inp2_unstable=121, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=40, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5155784, upper bound: 27.5164143
time: 9.33 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5155858, upper bound: 27.5164354
time: 6.02 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -18.9417896, 17.1321526, -21.2173862, 19.1514759, -38.0932655, 38.3495407
1: -18.5285168, 11.6903448, -20.6421242, 13.0044975, -31.5330124, 32.3324699
2: -22.1997299, 14.6061773, -24.8416786, 16.2886829, -38.4884109, 39.4478569
3: -26.4721489, 12.7498407, -29.6751900, 14.2242517, -40.6963959, 42.4250298
4: -24.1404076, 15.4883146, -27.0036907, 17.2983627, -41.4387703, 42.4920044
5: -18.4543190, 16.6426048, -20.6394119, 18.6154327, -37.0697441, 37.2820129
6: -19.6279545, 17.8056183, -21.9554615, 19.8948612, -39.5228157, 39.7610779
7: -23.5626316, 17.1840858, -26.3355846, 19.1797428, -42.7423706, 43.5196686
8: -28.4645538, 14.2836647, -31.8518085, 15.9549694, -44.4195251, 46.1354752
9: -17.1002083, 19.3055019, -19.1391087, 21.5745583, -38.6747665, 38.4446106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=119, inp2_unstable=121, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=41, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 197

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5165552, upper bound: 27.5166218
time: 9.56 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5165605, upper bound: 27.5166339
time: 7.04 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -18.7118073, 16.9442081, -19.6070557, 17.7238979, -36.4357033, 36.5512619
1: -18.3230305, 11.5076675, -19.1552505, 12.0838556, -30.4068871, 30.6629181
2: -21.9243584, 14.4351387, -22.9607162, 15.1082869, -37.0326462, 37.3958549
3: -26.1903458, 12.5908737, -27.3889446, 13.1845560, -39.3749008, 39.9798203
4: -23.8958435, 15.2894421, -24.9647827, 16.0257359, -39.9215775, 40.2542267
5: -18.2457371, 16.4590759, -19.0884285, 17.2261391, -35.4718742, 35.5475006
6: -19.4220123, 17.5883694, -20.3049183, 18.4096260, -37.8316383, 37.8932877
7: -23.3283329, 16.9784679, -24.3519306, 17.7632713, -41.0916061, 41.3303986
8: -28.1765041, 14.0732365, -29.4665985, 14.7996998, -42.9762039, 43.5398254
9: -16.8795528, 19.0873051, -17.6965828, 19.9632912, -36.8428421, 36.7838821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=115, inp2_unstable=119, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=39, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 197

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5161373, upper bound: 27.5175032
time: 13.01 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5161373, upper bound: 27.5176149
time: 6.08 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -19.5605583, 17.6798630, -19.9909763, 18.0556717, -37.6162224, 37.6708298
1: -19.1119938, 12.0677233, -19.5117416, 12.3348780, -31.4468689, 31.5794640
2: -22.9253712, 15.0717449, -23.4123039, 15.3949966, -38.3203659, 38.4840469
3: -27.3188877, 13.1540470, -27.8964157, 13.4385204, -40.7574081, 41.0504608
4: -24.9087791, 15.9915171, -25.4217548, 16.3409786, -41.2497559, 41.4132690
5: -19.0498371, 17.1751194, -19.4525185, 17.5475006, -36.5973358, 36.6276398
6: -20.2557373, 18.3764114, -20.6807880, 18.7663364, -39.0220718, 39.0571976
7: -24.3085480, 17.7292099, -24.7945366, 18.1003685, -42.4089127, 42.5237465
8: -29.3779640, 14.7585392, -30.0080490, 15.1063385, -44.4842987, 44.7665863
9: -17.6563110, 19.9197960, -18.0453281, 20.3376846, -37.9939880, 37.9651184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=119, inp2_unstable=120, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5161373, upper bound: 27.5176898
time: 6.40 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5170138, upper bound: 27.5177575
time: 5.92 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -18.5745220, 16.8262730, -21.0201454, 18.9853458, -37.5598679, 37.8464203
1: -18.1927071, 11.4150724, -20.4608116, 12.8684998, -31.0612068, 31.8758850
2: -21.7620869, 14.3317699, -24.6090126, 16.1438904, -37.9059715, 38.9407806
3: -26.0096436, 12.5013847, -29.4221554, 14.0938988, -40.1035423, 41.9235382
4: -23.7323704, 15.1760292, -26.7775650, 17.1368256, -40.8691902, 41.9535904
5: -18.1147938, 16.3449440, -20.4560432, 18.4555550, -36.5703506, 36.8009834
6: -19.2852077, 17.4614182, -21.7700233, 19.7110653, -38.9962730, 39.2314415
7: -23.1681900, 16.8574848, -26.1189270, 19.0083675, -42.1765594, 42.9764099
8: -27.9849625, 13.9626846, -31.5862541, 15.7953815, -43.7803383, 45.5489388
9: -16.7547150, 18.9515247, -18.9600754, 21.3854141, -38.1401291, 37.9115982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=115, inp2_unstable=121, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=40, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5161255, upper bound: 27.5167854
time: 5.17 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5161255, upper bound: 27.5168614
time: 10.07 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -19.4110355, 17.5526180, -21.4228649, 19.3327084, -38.7437439, 38.9754829
1: -18.9719620, 11.9666681, -20.8357601, 13.1303749, -32.1023369, 32.8024292
2: -22.7499733, 14.9601650, -25.0830326, 16.4430561, -39.1930237, 40.0431900
3: -27.1280479, 13.0569935, -29.9558105, 14.3583813, -41.4864159, 43.0128021
4: -24.7337036, 15.8682165, -27.2577686, 17.4655399, -42.1992416, 43.1259842
5: -18.9083652, 17.0529118, -20.8369350, 18.7918911, -37.7002525, 37.8898468
6: -20.1104813, 18.2375565, -22.1639748, 20.0841770, -40.1946564, 40.4015312
7: -24.1378784, 17.5993652, -26.5832405, 19.3609505, -43.4988289, 44.1826019
8: -29.1723137, 14.6384592, -32.1539803, 16.1141796, -45.2864914, 46.7924385
9: -17.5219135, 19.7757664, -19.3248177, 21.7780762, -39.2999878, 39.1005783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=119, inp2_unstable=121, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=41, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 197

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5170008, upper bound: 27.5169406
time: 10.31 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5170008, upper bound: 27.5170000
time: 9.11 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 20.87 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.87
Output dim: 1, lower bound: -27.5160050, upper bound: 27.5174886
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.87
Output dim: 1, lower bound: -27.5160050, upper bound: 27.5175559
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.87
Output dim: 1, lower bound: -27.5169536, upper bound: 27.5176896
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.87
Output dim: 1, lower bound: -27.5169536, upper bound: 27.5177485
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.87
Output dim: 1, lower bound: -27.5155784, upper bound: 27.5164143
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.87
Output dim: 1, lower bound: -27.5155858, upper bound: 27.5164354
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.87
Output dim: 1, lower bound: -27.5165552, upper bound: 27.5166218
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.87
Output dim: 1, lower bound: -27.5165605, upper bound: 27.5166339
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.87
Output dim: 1, lower bound: -27.5161373, upper bound: 27.5175032
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.87
Output dim: 1, lower bound: -27.5161373, upper bound: 27.5176149
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.87
Output dim: 1, lower bound: -27.5161373, upper bound: 27.5176898
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.87
Output dim: 1, lower bound: -27.5170138, upper bound: 27.5177575
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.87
Output dim: 1, lower bound: -27.5161255, upper bound: 27.5167854
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.87
Output dim: 1, lower bound: -27.5161255, upper bound: 27.5168614
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.87
Output dim: 1, lower bound: -27.5170008, upper bound: 27.5169406
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.87
Output dim: 1, lower bound: -27.5170008, upper bound: 27.5170000

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -18.2013054, 16.4866562, -18.6158905, 16.8410110, -35.0423164, 35.1025467
1: -17.8407879, 11.2107658, -18.2206249, 11.4911451, -29.3319283, 29.4313889
2: -21.3263588, 14.0511751, -21.7973289, 14.3604527, -35.6868134, 35.8485031
3: -25.4728603, 12.2558260, -26.0148201, 12.5367966, -38.0096588, 38.2706413
4: -23.2468185, 14.8770647, -23.7237663, 15.2218409, -38.4686584, 38.6008301
5: -17.7513828, 16.0112190, -18.1312408, 16.3652363, -34.1166153, 34.1424599
6: -18.8941250, 17.1186275, -19.2894497, 17.4978886, -36.3920135, 36.4080734
7: -22.6991920, 16.5269737, -23.1438789, 16.8871937, -39.5863876, 39.6708527
8: -27.4064026, 13.6893997, -27.9844322, 14.0419493, -41.4483490, 41.6738319
9: -16.4214363, 18.5750656, -16.8035412, 18.9736900, -35.3951225, 35.3786011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=54, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=115, inp2_unstable=119, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5160050, upper bound: 27.5174886
time: 6.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5160050, upper bound: 27.5174886
time: 7.14 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -18.2013054, 16.4866562, -19.0952835, 17.2706871, -35.4719925, 35.5819359
1: -17.8407879, 11.2107658, -18.6734734, 11.7719765, -29.6127567, 29.8842392
2: -21.3263588, 14.0511751, -22.3590717, 14.7223072, -36.0486603, 36.4102478
3: -25.4728603, 12.2558260, -26.6860771, 12.8506184, -38.3234787, 38.9418983
4: -23.2468185, 14.8770647, -24.3303642, 15.6096134, -38.8564301, 39.2074280
5: -17.7513828, 16.0112190, -18.5950317, 16.7852726, -34.5366554, 34.6062508
6: -18.8941250, 17.1186275, -19.7828522, 17.9383545, -36.8324776, 36.9014816
7: -22.6991920, 16.5269737, -23.7319412, 17.3110504, -40.0102425, 40.2589149
8: -27.4064026, 13.6893997, -28.7091694, 14.4043741, -41.8107758, 42.3985672
9: -16.4214363, 18.5750656, -17.2339954, 19.4546833, -35.8761215, 35.8090591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=54, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=115, inp2_unstable=119, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 197

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5160050, upper bound: 27.5175559
time: 12.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5160050, upper bound: 27.5175559
time: 8.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -19.0918045, 17.2597313, -18.9898472, 17.1644859, -36.2562904, 36.2495728
1: -18.6689243, 11.7919111, -18.5676212, 11.7357683, -30.4046936, 30.3595314
2: -22.3758392, 14.7180252, -22.2376747, 14.6404934, -37.0163345, 36.9556923
3: -26.6633892, 12.8471889, -26.5099735, 12.7841358, -39.4475212, 39.3571587
4: -24.3159409, 15.6121664, -24.1691742, 15.5300074, -39.8459473, 39.7813416
5: -18.5963440, 16.7650948, -18.4860668, 16.6791344, -35.2754784, 35.2511597
6: -19.7736626, 17.9449997, -19.6562595, 17.8445606, -37.6182251, 37.6012573
7: -23.7339211, 17.3144455, -23.5757256, 17.2158833, -40.9498024, 40.8901672
8: -28.6704998, 14.4042587, -28.5131073, 14.3419533, -43.0124512, 42.9173660
9: -17.2351589, 19.4499187, -17.1439266, 19.3391953, -36.5743561, 36.5938454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=54, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=119, inp2_unstable=120, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5169536, upper bound: 27.5176896
time: 6.25 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5169536, upper bound: 27.5176896
time: 8.53 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -19.0918045, 17.2597313, -19.4763374, 17.6002426, -36.6920471, 36.7360611
1: -18.6689243, 11.7919111, -19.0275402, 12.0211391, -30.6900616, 30.8194504
2: -22.3758392, 14.7180252, -22.8073807, 15.0071564, -37.3829956, 37.5254059
3: -26.6633892, 12.8471889, -27.1909790, 13.1027355, -39.7661209, 40.0381622
4: -24.3159409, 15.6121664, -24.7842712, 15.9229813, -40.2389145, 40.3964386
5: -18.5963440, 16.7650948, -18.9566078, 17.1048412, -35.7011795, 35.7217026
6: -19.7736626, 17.9449997, -20.1564980, 18.2918720, -38.0655327, 38.1014977
7: -23.7339211, 17.3144455, -24.1719551, 17.6458797, -41.3797913, 41.4863968
8: -28.6704998, 14.4042587, -29.2472782, 14.7090225, -43.3795242, 43.6515350
9: -17.2351589, 19.4499187, -17.5805073, 19.8269577, -37.0621109, 37.0304222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=54, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=119, inp2_unstable=120, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5169536, upper bound: 27.5177486
time: 14.26 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5169536, upper bound: 27.5177485
time: 12.45 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -17.7184277, 16.0641365, -19.0643806, 17.2718620, -34.9902802, 35.1285172
1: -17.3863316, 10.8964195, -18.6306934, 11.6206160, -29.0069447, 29.5271091
2: -20.7430782, 13.6838417, -22.2641239, 14.6688309, -35.4119110, 35.9479599
3: -24.8154335, 11.9351749, -26.7578049, 12.8064442, -37.6218796, 38.6929779
4: -22.6579628, 14.4803858, -24.3828144, 15.5373344, -38.1952972, 38.8631935
5: -17.2833042, 15.6030731, -18.5633526, 16.8002510, -34.0835495, 34.1664238
6: -18.4042511, 16.6652527, -19.7943401, 17.8820534, -36.2863045, 36.4595909
7: -22.1201744, 16.0944519, -23.7727318, 17.2693882, -39.3895645, 39.8671837
8: -26.7177982, 13.3051033, -28.7806473, 14.2587404, -40.9765282, 42.0857506
9: -15.9772358, 18.0900116, -17.1710949, 19.4338455, -35.4110756, 35.2611008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=54, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=115, inp2_unstable=120, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5151246, upper bound: 27.5161569
time: 8.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5152463, upper bound: 27.5161921
time: 7.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -17.8596611, 16.1883717, -19.8188286, 17.9305916, -35.7902527, 36.0071945
1: -17.5186310, 10.9857750, -19.3396263, 12.0951777, -29.6138058, 30.3254013
2: -20.9134560, 13.7911739, -23.1536598, 15.2310200, -36.1444740, 36.9448280
3: -25.0111408, 12.0295448, -27.7956161, 13.3000183, -38.3111572, 39.8251610
4: -22.8313923, 14.5946217, -25.3103867, 16.1456394, -38.9770317, 39.9050064
5: -17.4205589, 15.7231579, -19.2894554, 17.4404202, -34.8609734, 35.0126114
6: -18.5478172, 16.7980461, -20.5586586, 18.5819378, -37.1297531, 37.3567009
7: -22.2897034, 16.2203197, -24.6742001, 17.9335098, -40.2232132, 40.8945198
8: -26.9207420, 13.4160538, -29.8613319, 14.8425980, -41.7633400, 43.2773857
9: -16.1063480, 18.2319279, -17.8545113, 20.1876373, -36.2939835, 36.0864372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=54, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=115, inp2_unstable=121, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=37, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5141190, upper bound: 27.5131627
time: 8.11 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5112589, upper bound: 27.5129183
time: 6.99 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -18.5757885, 16.8124828, -19.4473495, 17.6064548, -36.1822395, 36.2598267
1: -18.1885147, 11.4539070, -18.9898186, 11.8693609, -30.0578766, 30.4437256
2: -21.7562008, 14.3274240, -22.7161388, 14.9543238, -36.7105255, 37.0435600
3: -25.9791489, 12.5078030, -27.2778969, 13.0605125, -39.0396614, 39.7856979
4: -23.6966457, 15.1863756, -24.8459320, 15.8507957, -39.5474396, 40.0323029
5: -18.0990143, 16.3353310, -18.9279270, 17.1266251, -35.2256393, 35.2632523
6: -19.2607002, 17.4602566, -20.1754742, 18.2353935, -37.4960899, 37.6357307
7: -23.1265335, 16.8563595, -24.2227039, 17.6075954, -40.7341309, 41.0790634
8: -27.9445438, 13.9928923, -29.3305168, 14.5641499, -42.5086784, 43.3234100
9: -16.7627354, 18.9420719, -17.5205765, 19.8150749, -36.5778046, 36.4626427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=54, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=119, inp2_unstable=120, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5165552, upper bound: 27.5166219
time: 7.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5165552, upper bound: 27.5166218
time: 6.55 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -18.7257309, 16.9433937, -20.2090759, 18.2704887, -36.9962158, 37.1524696
1: -18.3279610, 11.5500183, -19.7054405, 12.3493586, -30.6773186, 31.2554588
2: -21.9366684, 14.4411802, -23.6143341, 15.5220747, -37.4587402, 38.0555153
3: -26.1828728, 12.6067877, -28.3217659, 13.5579033, -39.7407761, 40.9285545
4: -23.8783970, 15.3095093, -25.7794495, 16.4664402, -40.3448334, 41.0889549
5: -18.2443523, 16.4617901, -19.6605587, 17.7714863, -36.0158386, 36.1223488
6: -19.4112816, 17.6006699, -20.9452362, 18.9416637, -38.3529434, 38.5459061
7: -23.3046265, 16.9899864, -25.1309509, 18.2774162, -41.5820427, 42.1209373
8: -28.1574364, 14.1119175, -30.4181099, 15.1545916, -43.3120270, 44.5300293
9: -16.9007874, 19.0910816, -18.2108231, 20.5738163, -37.4746017, 37.3019028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=54, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=119, inp2_unstable=121, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5165605, upper bound: 27.5166340
time: 6.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5165605, upper bound: 27.5166343
time: 7.13 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -18.7118073, 16.9442081, -18.6158905, 16.8410110, -35.5528183, 35.5600967
1: -18.3230305, 11.5076675, -18.2206249, 11.4911451, -29.8141747, 29.7282925
2: -21.9243584, 14.4351387, -21.7973289, 14.3604527, -36.2848129, 36.2324600
3: -26.1903458, 12.5908737, -26.0148201, 12.5367966, -38.7271423, 38.6056938
4: -23.8958435, 15.2894421, -23.7237663, 15.2218409, -39.1176834, 39.0132065
5: -18.2457371, 16.4590759, -18.1312408, 16.3652363, -34.6109695, 34.5903168
6: -19.4220123, 17.5883694, -19.2894497, 17.4978886, -36.9198990, 36.8778191
7: -23.3283329, 16.9784679, -23.1438789, 16.8871937, -40.2155228, 40.1223412
8: -28.1765041, 14.0732365, -27.9844322, 14.0419493, -42.2184525, 42.0576668
9: -16.8795528, 19.0873051, -16.8035412, 18.9736900, -35.8532410, 35.8908463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=54, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=115, inp2_unstable=119, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5161373, upper bound: 27.5175032
time: 6.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5161373, upper bound: 27.5175031
time: 7.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -18.7118073, 16.9442081, -19.0952835, 17.2706871, -35.9824905, 36.0394859
1: -18.3230305, 11.5076675, -18.6734734, 11.7719765, -30.0950069, 30.1811409
2: -21.9243584, 14.4351387, -22.3590717, 14.7223072, -36.6466675, 36.7942047
3: -26.1903458, 12.5908737, -26.6860771, 12.8506184, -39.0409622, 39.2769508
4: -23.8958435, 15.2894421, -24.3303642, 15.6096134, -39.5054550, 39.6198044
5: -18.2457371, 16.4590759, -18.5950317, 16.7852726, -35.0310097, 35.0541039
6: -19.4220123, 17.5883694, -19.7828522, 17.9383545, -37.3603630, 37.3712196
7: -23.3283329, 16.9784679, -23.7319412, 17.3110504, -40.6393814, 40.7104034
8: -28.1765041, 14.0732365, -28.7091694, 14.4043741, -42.5808792, 42.7823982
9: -16.8795528, 19.0873051, -17.2339954, 19.4546833, -36.3342361, 36.3213005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=54, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=115, inp2_unstable=119, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5161373, upper bound: 27.5176149
time: 5.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5161373, upper bound: 27.5176149
time: 6.08 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -19.5605583, 17.6798630, -18.9898472, 17.1644859, -36.7250443, 36.6697044
1: -19.1119938, 12.0677233, -18.5676212, 11.7357683, -30.8477631, 30.6353397
2: -22.9253712, 15.0717449, -22.2376747, 14.6404934, -37.5658646, 37.3094177
3: -27.3188877, 13.1540470, -26.5099735, 12.7841358, -40.1030235, 39.6640205
4: -24.9087791, 15.9915171, -24.1691742, 15.5300074, -40.4387856, 40.1606903
5: -19.0498371, 17.1751194, -18.4860668, 16.6791344, -35.7289734, 35.6611862
6: -20.2557373, 18.3764114, -19.6562595, 17.8445606, -38.1002960, 38.0326691
7: -24.3085480, 17.7292099, -23.5757256, 17.2158833, -41.5244293, 41.3049278
8: -29.3779640, 14.7585392, -28.5131073, 14.3419533, -43.7199097, 43.2716446
9: -17.6563110, 19.9197960, -17.1439266, 19.3391953, -36.9954987, 37.0637207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=54, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=119, inp2_unstable=120, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 197

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5170138, upper bound: 27.5176898
time: 6.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5170138, upper bound: 27.5176898
time: 8.56 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -19.5605583, 17.6798630, -19.4763374, 17.6002426, -37.1608009, 37.1561890
1: -19.1119938, 12.0677233, -19.0275402, 12.0211391, -31.1331329, 31.0952568
2: -22.9253712, 15.0717449, -22.8073807, 15.0071564, -37.9325218, 37.8791275
3: -27.3188877, 13.1540470, -27.1909790, 13.1027355, -40.4216232, 40.3450241
4: -24.9087791, 15.9915171, -24.7842712, 15.9229813, -40.8317528, 40.7757874
5: -19.0498371, 17.1751194, -18.9566078, 17.1048412, -36.1546707, 36.1317291
6: -20.2557373, 18.3764114, -20.1564980, 18.2918720, -38.5476036, 38.5329094
7: -24.3085480, 17.7292099, -24.1719551, 17.6458797, -41.9544220, 41.9011574
8: -29.3779640, 14.7585392, -29.2472782, 14.7090225, -44.0869751, 44.0058174
9: -17.6563110, 19.9197960, -17.5805073, 19.8269577, -37.4832611, 37.5003014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=54, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=119, inp2_unstable=120, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 197

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5170138, upper bound: 27.5177575
time: 8.60 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5170138, upper bound: 27.5177575
time: 5.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -18.5745220, 16.8262730, -20.0635376, 18.1342754, -36.7087936, 36.8898087
1: -18.1927071, 11.4150724, -19.5578957, 12.2920799, -30.4847870, 30.9729691
2: -21.7620869, 14.3317699, -23.4846573, 15.4223032, -37.1843910, 37.8164291
3: -26.0096436, 12.5013847, -28.1001110, 13.4679089, -39.4775543, 40.6014938
4: -23.7323704, 15.1760292, -25.5826302, 16.3603344, -40.0927048, 40.7586555
5: -18.1147938, 16.3449440, -19.5323658, 17.6265869, -35.7413788, 35.8773117
6: -19.2852077, 17.4614182, -20.7919102, 18.8291950, -38.1144028, 38.2533226
7: -23.1681900, 16.8574848, -24.9539337, 18.1621246, -41.3303146, 41.8114166
8: -27.9849625, 13.9626846, -30.1590996, 15.0623198, -43.0472832, 44.1217842
9: -16.7547150, 18.9515247, -18.0977631, 20.4321327, -37.1868477, 37.0492859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=54, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=115, inp2_unstable=121, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=39, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 197

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5145827, upper bound: 27.5135392
time: 29.34 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5119995, upper bound: 27.5133365
time: 5.93 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 36.73 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5160050, upper bound: 27.5174886
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5160050, upper bound: 27.5174886
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5160050, upper bound: 27.5175559
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5160050, upper bound: 27.5175559
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5169536, upper bound: 27.5176896
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5169536, upper bound: 27.5176896
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5169536, upper bound: 27.5177486
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5169536, upper bound: 27.5177485
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5151246, upper bound: 27.5161569
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5152463, upper bound: 27.5161921
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5141190, upper bound: 27.5131627
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5112589, upper bound: 27.5129183
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5165552, upper bound: 27.5166219
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5165552, upper bound: 27.5166218
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5165605, upper bound: 27.5166340
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5165605, upper bound: 27.5166343
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5161373, upper bound: 27.5175032
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5161373, upper bound: 27.5175031
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5161373, upper bound: 27.5176149
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5161373, upper bound: 27.5176149
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5170138, upper bound: 27.5176898
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5170138, upper bound: 27.5176898
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5170138, upper bound: 27.5177575
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5170138, upper bound: 27.5177575
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5145827, upper bound: 27.5135392
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 36.73
Output dim: 1, lower bound: -27.5119995, upper bound: 27.5133365
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 36.73
Output dim: 1, lower bound: -27.5161255, upper bound: 27.5168614
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 36.73
Output dim: 1, lower bound: -27.5170008, upper bound: 27.5169406
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 36.73
Output dim: 1, lower bound: -27.5170008, upper bound: 27.5170000

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 17.05 + 586.98 = 604.04 seconds
