## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.001979395


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078547, 0.0078547)
1: (-0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022145, 0.0022145)
2: (-0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0163393, 0.0163393)
3: (-0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021622, 0.0021622)
4: (0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0122110, 0.0122110)
5: (0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033926, 0.0033926)
6: (0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030794, 0.0030794)
7: (-0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114919, 0.0114919)
8: (-0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0089442, 0.0089442)
9: (-0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007717, 0.0007717)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.17 + 2.82 = 4.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0023286, upper bound: 0.0023287

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022467, upper bound: 0.0022188
time: 1.56 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022467, upper bound: 0.0022467
time: 1.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.57 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.57
Output dim: 5, lower bound: -0.0022467, upper bound: 0.0022188
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.57
Output dim: 5, lower bound: -0.0022467, upper bound: 0.0022467

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0152866, -0.0061908, -0.0155945, -0.0061038, -0.0074385, 0.0073441
1: -0.0072485, -0.0046841, -0.0073353, -0.0046596, -0.0020972, 0.0020706
2: -0.0149214, 0.0039997, -0.0155618, 0.0041807, -0.0154736, 0.0152772
3: -0.0003473, 0.0021566, -0.0004321, 0.0021805, -0.0020477, 0.0020217
4: 0.0031027, 0.0172432, 0.0029675, 0.0177218, -0.0114172, 0.0115640
5: 0.9963683, 1.0002968, 0.9963307, 1.0004299, -0.0031720, 0.0032128
6: 0.0045871, 0.0081532, 0.0045530, 0.0082739, -0.0028793, 0.0029163
7: -0.0062631, 0.0070446, -0.0063904, 0.0074951, -0.0107449, 0.0108830
8: -0.0146757, -0.0043183, -0.0150263, -0.0042192, -0.0084703, 0.0083627
9: -0.0036372, -0.0027436, -0.0036457, -0.0027133, -0.0007215, 0.0007308

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022188, upper bound: 0.0022188
time: 1.88 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022188, upper bound: 0.0022187
time: 2.06 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0153697, -0.0059212, -0.0155960, -0.0061009, -0.0075136, 0.0080977
1: -0.0072719, -0.0046081, -0.0073357, -0.0046587, -0.0021184, 0.0022831
2: -0.0150942, 0.0045606, -0.0155649, 0.0041867, -0.0156297, 0.0168449
3: -0.0003702, 0.0022308, -0.0004325, 0.0021813, -0.0020683, 0.0022292
4: 0.0026835, 0.0173724, 0.0029630, 0.0177241, -0.0125888, 0.0116807
5: 0.9962518, 1.0003327, 0.9963294, 1.0004305, -0.0034976, 0.0032452
6: 0.0044814, 0.0081857, 0.0045519, 0.0082744, -0.0031747, 0.0029457
7: -0.0066576, 0.0071662, -0.0063947, 0.0074972, -0.0118475, 0.0109928
8: -0.0147703, -0.0040112, -0.0150280, -0.0042159, -0.0085557, 0.0092209
9: -0.0036637, -0.0027354, -0.0036460, -0.0027132, -0.0007955, 0.0007381

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022188, upper bound: 0.0022467
time: 1.95 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022188, upper bound: 0.0022467
time: 2.05 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 6.25 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 6.25
Output dim: 5, lower bound: -0.0022188, upper bound: 0.0022188
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 6.25
Output dim: 5, lower bound: -0.0022188, upper bound: 0.0022187
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 6.25
Output dim: 5, lower bound: -0.0022188, upper bound: 0.0022467
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 6.25
Output dim: 5, lower bound: -0.0022188, upper bound: 0.0022467

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0152866, -0.0061908, -0.0152866, -0.0061908, -0.0070579, 0.0070579
1: -0.0072485, -0.0046841, -0.0072485, -0.0046841, -0.0019899, 0.0019899
2: -0.0149214, 0.0039997, -0.0149214, 0.0039997, -0.0146818, 0.0146818
3: -0.0003473, 0.0021566, -0.0003473, 0.0021566, -0.0019429, 0.0019429
4: 0.0031027, 0.0172432, 0.0031027, 0.0172432, -0.0109722, 0.0109722
5: 0.9963683, 1.0002968, 0.9963683, 1.0002968, -0.0030484, 0.0030484
6: 0.0045871, 0.0081532, 0.0045871, 0.0081532, -0.0027670, 0.0027670
7: -0.0062631, 0.0070446, -0.0062631, 0.0070446, -0.0103261, 0.0103261
8: -0.0146757, -0.0043183, -0.0146757, -0.0043183, -0.0080368, 0.0080368
9: -0.0036372, -0.0027436, -0.0036372, -0.0027436, -0.0006934, 0.0006934

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021889, upper bound: 0.0021626
time: 1.56 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021888, upper bound: 0.0021889
time: 1.48 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0152866, -0.0061908, -0.0153697, -0.0059212, -0.0077822, 0.0072402
1: -0.0072485, -0.0046841, -0.0072719, -0.0046081, -0.0021941, 0.0020413
2: -0.0149214, 0.0039997, -0.0150942, 0.0045606, -0.0161886, 0.0150612
3: -0.0003473, 0.0021566, -0.0003702, 0.0022308, -0.0021423, 0.0019931
4: 0.0031027, 0.0172432, 0.0026835, 0.0173724, -0.0112558, 0.0120983
5: 0.9963683, 1.0002968, 0.9962518, 1.0003327, -0.0031272, 0.0033613
6: 0.0045871, 0.0081532, 0.0044814, 0.0081857, -0.0028385, 0.0030510
7: -0.0062631, 0.0070446, -0.0066576, 0.0071662, -0.0105930, 0.0113859
8: -0.0146757, -0.0043183, -0.0147703, -0.0040112, -0.0088617, 0.0082445
9: -0.0036372, -0.0027436, -0.0036637, -0.0027354, -0.0007113, 0.0007645

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021889, upper bound: 0.0021626
time: 2.00 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021888, upper bound: 0.0021889
time: 2.20 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0153697, -0.0059212, -0.0152866, -0.0061908, -0.0072402, 0.0077822
1: -0.0072719, -0.0046081, -0.0072485, -0.0046841, -0.0020413, 0.0021941
2: -0.0150942, 0.0045606, -0.0149214, 0.0039997, -0.0150612, 0.0161886
3: -0.0003702, 0.0022308, -0.0003473, 0.0021566, -0.0019931, 0.0021423
4: 0.0026835, 0.0173724, 0.0031027, 0.0172432, -0.0120983, 0.0112558
5: 0.9962518, 1.0003327, 0.9963683, 1.0002968, -0.0033613, 0.0031272
6: 0.0044814, 0.0081857, 0.0045871, 0.0081532, -0.0030510, 0.0028385
7: -0.0066576, 0.0071662, -0.0062631, 0.0070446, -0.0113859, 0.0105930
8: -0.0147703, -0.0040112, -0.0146757, -0.0043183, -0.0082445, 0.0088617
9: -0.0036637, -0.0027354, -0.0036372, -0.0027436, -0.0007645, 0.0007113

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021888, upper bound: 0.0021925
time: 2.05 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021887, upper bound: 0.0022156
time: 2.06 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0153697, -0.0059212, -0.0153697, -0.0059212, -0.0075718, 0.0075718
1: -0.0072719, -0.0046081, -0.0072719, -0.0046081, -0.0021348, 0.0021348
2: -0.0150942, 0.0045606, -0.0150942, 0.0045606, -0.0157508, 0.0157508
3: -0.0003702, 0.0022308, -0.0003702, 0.0022308, -0.0020844, 0.0020844
4: 0.0026835, 0.0173724, 0.0026835, 0.0173724, -0.0117712, 0.0117712
5: 0.9962518, 1.0003327, 0.9962518, 1.0003327, -0.0032704, 0.0032704
6: 0.0044814, 0.0081857, 0.0044814, 0.0081857, -0.0029685, 0.0029685
7: -0.0066576, 0.0071662, -0.0066576, 0.0071662, -0.0110780, 0.0110780
8: -0.0147703, -0.0040112, -0.0147703, -0.0040112, -0.0086220, 0.0086220
9: -0.0036637, -0.0027354, -0.0036637, -0.0027354, -0.0007439, 0.0007439

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021888, upper bound: 0.0021926
time: 2.09 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021888, upper bound: 0.0022156
time: 2.07 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.48 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.48
Output dim: 5, lower bound: -0.0021889, upper bound: 0.0021626
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.48
Output dim: 5, lower bound: -0.0021888, upper bound: 0.0021889
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.48
Output dim: 5, lower bound: -0.0021889, upper bound: 0.0021626
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.48
Output dim: 5, lower bound: -0.0021888, upper bound: 0.0021889
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.48
Output dim: 5, lower bound: -0.0021888, upper bound: 0.0021925
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.48
Output dim: 5, lower bound: -0.0021887, upper bound: 0.0022156
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.48
Output dim: 5, lower bound: -0.0021888, upper bound: 0.0021926
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.48
Output dim: 5, lower bound: -0.0021888, upper bound: 0.0022156

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0150751, -0.0062721, -0.0152318, -0.0062112, -0.0068930, 0.0069319
1: -0.0071889, -0.0047070, -0.0072331, -0.0046898, -0.0019434, 0.0019544
2: -0.0144813, 0.0038306, -0.0148074, 0.0039573, -0.0143388, 0.0144198
3: -0.0002891, 0.0021342, -0.0003322, 0.0021510, -0.0018975, 0.0019082
4: 0.0032291, 0.0169143, 0.0031344, 0.0171580, -0.0107764, 0.0107159
5: 0.9964034, 1.0002056, 0.9963771, 1.0002732, -0.0029940, 0.0029772
6: 0.0046190, 0.0080702, 0.0045951, 0.0081317, -0.0027177, 0.0027024
7: -0.0061442, 0.0067351, -0.0062333, 0.0069644, -0.0101418, 0.0100849
8: -0.0144348, -0.0044109, -0.0146133, -0.0043415, -0.0078491, 0.0078934
9: -0.0036292, -0.0027644, -0.0036352, -0.0027490, -0.0006810, 0.0006772

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0021024
time: 1.39 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021556, upper bound: 0.0021286
time: 1.42 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0151588, -0.0061351, -0.0152516, -0.0062073, -0.0068810, 0.0071178
1: -0.0072125, -0.0046684, -0.0072387, -0.0046887, -0.0019400, 0.0020068
2: -0.0146556, 0.0041156, -0.0148487, 0.0039655, -0.0143139, 0.0148065
3: -0.0003121, 0.0021719, -0.0003377, 0.0021521, -0.0018942, 0.0019594
4: 0.0030161, 0.0170445, 0.0031283, 0.0171888, -0.0110654, 0.0106973
5: 0.9963443, 1.0002418, 0.9963754, 1.0002818, -0.0030743, 0.0029720
6: 0.0045653, 0.0081031, 0.0045936, 0.0081394, -0.0027905, 0.0026977
7: -0.0063446, 0.0068577, -0.0062391, 0.0069935, -0.0104138, 0.0100673
8: -0.0145302, -0.0042548, -0.0146359, -0.0043370, -0.0078354, 0.0081051
9: -0.0036426, -0.0027561, -0.0036356, -0.0027470, -0.0006993, 0.0006760

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0021286
time: 1.92 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021555, upper bound: 0.0021556
time: 1.85 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0150751, -0.0062721, -0.0153179, -0.0059422, -0.0072939, 0.0071139
1: -0.0071889, -0.0047070, -0.0072573, -0.0046140, -0.0020564, 0.0020057
2: -0.0144813, 0.0038306, -0.0149865, 0.0045169, -0.0151727, 0.0147984
3: -0.0002891, 0.0021342, -0.0003559, 0.0022250, -0.0020079, 0.0019583
4: 0.0032291, 0.0169143, 0.0027163, 0.0172918, -0.0110594, 0.0113391
5: 0.9964034, 1.0002056, 0.9962609, 1.0003104, -0.0030726, 0.0031503
6: 0.0046190, 0.0080702, 0.0044897, 0.0081654, -0.0027890, 0.0028596
7: -0.0061442, 0.0067351, -0.0066268, 0.0070904, -0.0104081, 0.0106714
8: -0.0144348, -0.0044109, -0.0147113, -0.0040352, -0.0083056, 0.0081007
9: -0.0036292, -0.0027644, -0.0036616, -0.0027405, -0.0006989, 0.0007166

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020577, upper bound: 0.0020996
time: 1.84 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021785, upper bound: 0.0021261
time: 1.44 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0151588, -0.0061351, -0.0153316, -0.0059383, -0.0072863, 0.0073000
1: -0.0072125, -0.0046684, -0.0072612, -0.0046129, -0.0020543, 0.0020581
2: -0.0146556, 0.0041156, -0.0150150, 0.0045250, -0.0151569, 0.0151854
3: -0.0003121, 0.0021719, -0.0003597, 0.0022261, -0.0020058, 0.0020095
4: 0.0030161, 0.0170445, 0.0027102, 0.0173131, -0.0113486, 0.0113274
5: 0.9963443, 1.0002418, 0.9962592, 1.0003164, -0.0031530, 0.0031471
6: 0.0045653, 0.0081031, 0.0044881, 0.0081708, -0.0028620, 0.0028566
7: -0.0063446, 0.0068577, -0.0066326, 0.0071104, -0.0106803, 0.0106603
8: -0.0145302, -0.0042548, -0.0147269, -0.0040307, -0.0082969, 0.0083125
9: -0.0036426, -0.0027561, -0.0036620, -0.0027392, -0.0007172, 0.0007158

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020576, upper bound: 0.0021272
time: 1.46 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021786, upper bound: 0.0021536
time: 1.34 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0151673, -0.0060016, -0.0152318, -0.0062112, -0.0070042, 0.0073396
1: -0.0072149, -0.0046307, -0.0072331, -0.0046898, -0.0019747, 0.0020693
2: -0.0146732, 0.0043933, -0.0148074, 0.0039573, -0.0145701, 0.0152679
3: -0.0003145, 0.0022087, -0.0003322, 0.0021510, -0.0019281, 0.0020205
4: 0.0028086, 0.0170577, 0.0031344, 0.0171580, -0.0114103, 0.0108888
5: 0.9962866, 1.0002453, 0.9963771, 1.0002732, -0.0031701, 0.0030252
6: 0.0045130, 0.0081064, 0.0045951, 0.0081317, -0.0028775, 0.0027460
7: -0.0065399, 0.0068700, -0.0062333, 0.0069644, -0.0107384, 0.0102476
8: -0.0145398, -0.0041028, -0.0146133, -0.0043415, -0.0079757, 0.0083577
9: -0.0036558, -0.0027553, -0.0036352, -0.0027490, -0.0007211, 0.0006881

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021292
time: 1.91 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021536, upper bound: 0.0021552
time: 1.83 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0152332, -0.0058773, -0.0152516, -0.0062073, -0.0070620, 0.0075049
1: -0.0072335, -0.0045957, -0.0072387, -0.0046887, -0.0019910, 0.0021159
2: -0.0148104, 0.0046519, -0.0148487, 0.0039655, -0.0146904, 0.0156117
3: -0.0003326, 0.0022429, -0.0003377, 0.0021521, -0.0019440, 0.0020660
4: 0.0026153, 0.0171602, 0.0031283, 0.0171888, -0.0116672, 0.0109787
5: 0.9962329, 1.0002738, 0.9963754, 1.0002818, -0.0032415, 0.0030502
6: 0.0044642, 0.0081322, 0.0045936, 0.0081394, -0.0029423, 0.0027687
7: -0.0067218, 0.0069665, -0.0062391, 0.0069935, -0.0109801, 0.0103321
8: -0.0146149, -0.0039613, -0.0146359, -0.0043370, -0.0080415, 0.0085459
9: -0.0036680, -0.0027488, -0.0036356, -0.0027470, -0.0007373, 0.0006938

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021521
time: 1.48 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021535, upper bound: 0.0021787
time: 1.54 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0151673, -0.0060016, -0.0153179, -0.0059422, -0.0070020, 0.0071125
1: -0.0072149, -0.0046307, -0.0072573, -0.0046140, -0.0019741, 0.0020053
2: -0.0146732, 0.0043933, -0.0149865, 0.0045169, -0.0145656, 0.0147955
3: -0.0003145, 0.0022087, -0.0003559, 0.0022250, -0.0019275, 0.0019579
4: 0.0028086, 0.0170577, 0.0027163, 0.0172918, -0.0110572, 0.0108855
5: 0.9962866, 1.0002453, 0.9962609, 1.0003104, -0.0030720, 0.0030243
6: 0.0045130, 0.0081064, 0.0044897, 0.0081654, -0.0027885, 0.0027452
7: -0.0065399, 0.0068700, -0.0066268, 0.0070904, -0.0104061, 0.0102444
8: -0.0145398, -0.0041028, -0.0147113, -0.0040352, -0.0079733, 0.0080991
9: -0.0036558, -0.0027553, -0.0036616, -0.0027405, -0.0006987, 0.0006879

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021292
time: 1.90 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021533, upper bound: 0.0021552
time: 1.87 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0152332, -0.0058773, -0.0153316, -0.0059383, -0.0070562, 0.0072979
1: -0.0072335, -0.0045957, -0.0072612, -0.0046129, -0.0019894, 0.0020576
2: -0.0148104, 0.0046519, -0.0150150, 0.0045250, -0.0146783, 0.0151812
3: -0.0003326, 0.0022429, -0.0003597, 0.0022261, -0.0019424, 0.0020090
4: 0.0026153, 0.0171602, 0.0027102, 0.0173131, -0.0113455, 0.0109697
5: 0.9962329, 1.0002738, 0.9962592, 1.0003164, -0.0031521, 0.0030477
6: 0.0044642, 0.0081322, 0.0044881, 0.0081708, -0.0028612, 0.0027664
7: -0.0067218, 0.0069665, -0.0066326, 0.0071104, -0.0106774, 0.0103237
8: -0.0146149, -0.0039613, -0.0147269, -0.0040307, -0.0080349, 0.0083102
9: -0.0036680, -0.0027488, -0.0036620, -0.0027392, -0.0007170, 0.0006932

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021522
time: 1.46 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021534, upper bound: 0.0021786
time: 1.94 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.61 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0021024
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 5, lower bound: -0.0021556, upper bound: 0.0021286
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0021286
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 5, lower bound: -0.0021555, upper bound: 0.0021556
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 5, lower bound: -0.0020577, upper bound: 0.0020996
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 5, lower bound: -0.0021785, upper bound: 0.0021261
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 5, lower bound: -0.0020576, upper bound: 0.0021272
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 5, lower bound: -0.0021786, upper bound: 0.0021536
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021292
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 5, lower bound: -0.0021536, upper bound: 0.0021552
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021521
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 5, lower bound: -0.0021535, upper bound: 0.0021787
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021292
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 5, lower bound: -0.0021533, upper bound: 0.0021552
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021522
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 5, lower bound: -0.0021534, upper bound: 0.0021786

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0149213, -0.0062961, -0.0146761, -0.0061490, -0.0068265, 0.0063934
1: -0.0071455, -0.0047138, -0.0070764, -0.0046723, -0.0019246, 0.0018026
2: -0.0141614, 0.0037807, -0.0136515, 0.0040866, -0.0142004, 0.0132997
3: -0.0002467, 0.0021276, -0.0001793, 0.0021681, -0.0018792, 0.0017600
4: 0.0032664, 0.0166752, 0.0030378, 0.0162941, -0.0099393, 0.0106125
5: 0.9964138, 1.0001391, 0.9963502, 1.0000333, -0.0027614, 0.0029485
6: 0.0046284, 0.0080099, 0.0045707, 0.0079138, -0.0025066, 0.0026763
7: -0.0061091, 0.0065101, -0.0063243, 0.0061515, -0.0093540, 0.0099876
8: -0.0142597, -0.0044381, -0.0139806, -0.0042707, -0.0077733, 0.0072803
9: -0.0036268, -0.0027795, -0.0036413, -0.0028036, -0.0006281, 0.0006706

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0020142
time: 1.58 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0021024
time: 2.03 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0150751, -0.0062721, -0.0151616, -0.0062268, -0.0068579, 0.0067886
1: -0.0071889, -0.0047070, -0.0072133, -0.0046942, -0.0019335, 0.0019140
2: -0.0144813, 0.0038306, -0.0146613, 0.0039247, -0.0142659, 0.0141217
3: -0.0002891, 0.0021342, -0.0003129, 0.0021467, -0.0018879, 0.0018688
4: 0.0032291, 0.0169143, 0.0031588, 0.0170488, -0.0105537, 0.0106614
5: 0.9964034, 1.0002056, 0.9963838, 1.0002429, -0.0029321, 0.0029621
6: 0.0046190, 0.0080702, 0.0046013, 0.0081041, -0.0026615, 0.0026887
7: -0.0061442, 0.0067351, -0.0062104, 0.0068617, -0.0099322, 0.0100336
8: -0.0144348, -0.0044109, -0.0145333, -0.0043593, -0.0078092, 0.0077303
9: -0.0036292, -0.0027644, -0.0036336, -0.0027559, -0.0006669, 0.0006737

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021286, upper bound: 0.0020142
time: 1.49 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021286, upper bound: 0.0021285
time: 2.03 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0150044, -0.0061592, -0.0146946, -0.0061453, -0.0068091, 0.0065738
1: -0.0071690, -0.0046752, -0.0070816, -0.0046712, -0.0019197, 0.0018534
2: -0.0143344, 0.0040654, -0.0136898, 0.0040944, -0.0141642, 0.0136749
3: -0.0002696, 0.0021653, -0.0001843, 0.0021691, -0.0018744, 0.0018097
4: 0.0030536, 0.0168045, 0.0030320, 0.0163228, -0.0102198, 0.0105854
5: 0.9963546, 1.0001750, 0.9963486, 1.0000412, -0.0028394, 0.0029410
6: 0.0045748, 0.0080425, 0.0045693, 0.0079210, -0.0025773, 0.0026695
7: -0.0063093, 0.0066318, -0.0063297, 0.0061784, -0.0096179, 0.0099621
8: -0.0143544, -0.0042823, -0.0140015, -0.0042665, -0.0077535, 0.0074857
9: -0.0036403, -0.0027713, -0.0036416, -0.0028018, -0.0006458, 0.0006689

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0020299
time: 1.80 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0021287
time: 1.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0151588, -0.0061351, -0.0151813, -0.0062230, -0.0068526, 0.0069479
1: -0.0072125, -0.0046684, -0.0072188, -0.0046932, -0.0019320, 0.0019589
2: -0.0146556, 0.0041156, -0.0147023, 0.0039327, -0.0142548, 0.0144531
3: -0.0003121, 0.0021719, -0.0003183, 0.0021477, -0.0018864, 0.0019126
4: 0.0030161, 0.0170445, 0.0031528, 0.0170795, -0.0108013, 0.0106531
5: 0.9963443, 1.0002418, 0.9963822, 1.0002514, -0.0030009, 0.0029598
6: 0.0045653, 0.0081031, 0.0045998, 0.0081119, -0.0027239, 0.0026866
7: -0.0063446, 0.0068577, -0.0062160, 0.0068906, -0.0101653, 0.0100258
8: -0.0145302, -0.0042548, -0.0145558, -0.0043549, -0.0078031, 0.0079116
9: -0.0036426, -0.0027561, -0.0036340, -0.0027539, -0.0006826, 0.0006732

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021287, upper bound: 0.0020299
time: 1.39 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021287, upper bound: 0.0021556
time: 1.89 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0149213, -0.0062961, -0.0147800, -0.0058700, -0.0072676, 0.0066351
1: -0.0071455, -0.0047138, -0.0071057, -0.0045936, -0.0020490, 0.0018707
2: -0.0141614, 0.0037807, -0.0138675, 0.0046671, -0.0151182, 0.0138023
3: -0.0002467, 0.0021276, -0.0002079, 0.0022449, -0.0020006, 0.0018265
4: 0.0032664, 0.0166752, 0.0026040, 0.0164556, -0.0103150, 0.0112984
5: 0.9964138, 1.0001391, 0.9962296, 1.0000781, -0.0028658, 0.0031390
6: 0.0046284, 0.0080099, 0.0044614, 0.0079545, -0.0026013, 0.0028493
7: -0.0061091, 0.0065101, -0.0067325, 0.0063034, -0.0097076, 0.0106330
8: -0.0142597, -0.0044381, -0.0140988, -0.0039529, -0.0082757, 0.0075554
9: -0.0036268, -0.0027795, -0.0036687, -0.0027934, -0.0006518, 0.0007140

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020576, upper bound: 0.0020116
time: 1.55 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020576, upper bound: 0.0020997
time: 1.46 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0150751, -0.0062721, -0.0152442, -0.0059578, -0.0072597, 0.0070143
1: -0.0071889, -0.0047070, -0.0072366, -0.0046184, -0.0020468, 0.0019776
2: -0.0144813, 0.0038306, -0.0148332, 0.0044844, -0.0151016, 0.0145912
3: -0.0002891, 0.0021342, -0.0003356, 0.0022207, -0.0019985, 0.0019309
4: 0.0032291, 0.0169143, 0.0027405, 0.0171773, -0.0109045, 0.0112860
5: 0.9964034, 1.0002056, 0.9962677, 1.0002786, -0.0030296, 0.0031356
6: 0.0046190, 0.0080702, 0.0044958, 0.0081365, -0.0027500, 0.0028462
7: -0.0061442, 0.0067351, -0.0066040, 0.0069826, -0.0102624, 0.0106214
8: -0.0144348, -0.0044109, -0.0146274, -0.0040529, -0.0082666, 0.0079872
9: -0.0036292, -0.0027644, -0.0036601, -0.0027478, -0.0006891, 0.0007132

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021520, upper bound: 0.0020116
time: 1.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021520, upper bound: 0.0021262
time: 1.89 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0150044, -0.0061592, -0.0147938, -0.0058665, -0.0072569, 0.0068159
1: -0.0071690, -0.0046752, -0.0071096, -0.0045927, -0.0020460, 0.0019217
2: -0.0143344, 0.0040654, -0.0138962, 0.0046743, -0.0150958, 0.0141785
3: -0.0002696, 0.0021653, -0.0002116, 0.0022459, -0.0019977, 0.0018763
4: 0.0030536, 0.0168045, 0.0025986, 0.0164770, -0.0105961, 0.0112817
5: 0.9963546, 1.0001750, 0.9962282, 1.0000840, -0.0029439, 0.0031344
6: 0.0045748, 0.0080425, 0.0044600, 0.0079599, -0.0026722, 0.0028451
7: -0.0063093, 0.0066318, -0.0067375, 0.0063236, -0.0099722, 0.0106173
8: -0.0143544, -0.0042823, -0.0141145, -0.0039490, -0.0082635, 0.0077613
9: -0.0036403, -0.0027713, -0.0036690, -0.0027920, -0.0006696, 0.0007129

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020576, upper bound: 0.0020281
time: 1.91 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020576, upper bound: 0.0021272
time: 1.94 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0151588, -0.0061351, -0.0152592, -0.0059540, -0.0072583, 0.0071755
1: -0.0072125, -0.0046684, -0.0072408, -0.0046173, -0.0020464, 0.0020230
2: -0.0146556, 0.0041156, -0.0148645, 0.0044924, -0.0150987, 0.0149265
3: -0.0003121, 0.0021719, -0.0003398, 0.0022218, -0.0019981, 0.0019753
4: 0.0030161, 0.0170445, 0.0027346, 0.0172006, -0.0111551, 0.0112838
5: 0.9963443, 1.0002418, 0.9962659, 1.0002850, -0.0030992, 0.0031350
6: 0.0045653, 0.0081031, 0.0044943, 0.0081424, -0.0028132, 0.0028456
7: -0.0063446, 0.0068577, -0.0066096, 0.0070046, -0.0104982, 0.0106193
8: -0.0145302, -0.0042548, -0.0146445, -0.0040486, -0.0082650, 0.0081708
9: -0.0036426, -0.0027561, -0.0036604, -0.0027463, -0.0007049, 0.0007131

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021521, upper bound: 0.0020281
time: 1.85 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021521, upper bound: 0.0021536
time: 1.32 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0150178, -0.0060266, -0.0146761, -0.0061490, -0.0069505, 0.0067994
1: -0.0071727, -0.0046378, -0.0070764, -0.0046723, -0.0019596, 0.0019170
2: -0.0143622, 0.0043414, -0.0136515, 0.0040866, -0.0144584, 0.0141441
3: -0.0002733, 0.0022018, -0.0001793, 0.0021681, -0.0019133, 0.0018718
4: 0.0028474, 0.0168253, 0.0030378, 0.0162941, -0.0105705, 0.0108053
5: 0.9962974, 1.0001808, 0.9963502, 1.0000333, -0.0029368, 0.0030020
6: 0.0045227, 0.0080478, 0.0045707, 0.0079138, -0.0026657, 0.0027249
7: -0.0065034, 0.0066513, -0.0063243, 0.0061515, -0.0099480, 0.0101690
8: -0.0143696, -0.0041313, -0.0139806, -0.0042707, -0.0079145, 0.0077425
9: -0.0036533, -0.0027700, -0.0036413, -0.0028036, -0.0006680, 0.0006828

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0020461
time: 1.50 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021292
time: 1.99 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0151673, -0.0060016, -0.0151616, -0.0062268, -0.0069757, 0.0072427
1: -0.0072149, -0.0046307, -0.0072133, -0.0046942, -0.0019667, 0.0020420
2: -0.0146732, 0.0043933, -0.0146613, 0.0039247, -0.0145109, 0.0150662
3: -0.0003145, 0.0022087, -0.0003129, 0.0021467, -0.0019203, 0.0019938
4: 0.0028086, 0.0170577, 0.0031588, 0.0170488, -0.0112596, 0.0108446
5: 0.9962866, 1.0002453, 0.9963838, 1.0002429, -0.0031282, 0.0030129
6: 0.0045130, 0.0081064, 0.0046013, 0.0081041, -0.0028395, 0.0027348
7: -0.0065399, 0.0068700, -0.0062104, 0.0068617, -0.0105965, 0.0102059
8: -0.0145398, -0.0041028, -0.0145333, -0.0043593, -0.0079433, 0.0082473
9: -0.0036558, -0.0027553, -0.0036336, -0.0027559, -0.0007115, 0.0006853

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021271, upper bound: 0.0020461
time: 1.42 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021271, upper bound: 0.0021552
time: 2.00 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0150822, -0.0059019, -0.0146946, -0.0061453, -0.0070080, 0.0069598
1: -0.0071909, -0.0046026, -0.0070816, -0.0046712, -0.0019758, 0.0019622
2: -0.0144962, 0.0046006, -0.0136898, 0.0040944, -0.0145781, 0.0144777
3: -0.0002910, 0.0022361, -0.0001843, 0.0021691, -0.0019292, 0.0019159
4: 0.0026537, 0.0169254, 0.0030320, 0.0163228, -0.0108197, 0.0108947
5: 0.9962435, 1.0002086, 0.9963486, 1.0000412, -0.0030060, 0.0030269
6: 0.0044739, 0.0080730, 0.0045693, 0.0079210, -0.0027286, 0.0027475
7: -0.0066858, 0.0067456, -0.0063297, 0.0061784, -0.0101826, 0.0102532
8: -0.0144430, -0.0039893, -0.0140015, -0.0042665, -0.0079800, 0.0079251
9: -0.0036656, -0.0027637, -0.0036416, -0.0028018, -0.0006837, 0.0006885

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020282, upper bound: 0.0020576
time: 1.62 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020282, upper bound: 0.0021522
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0152332, -0.0058773, -0.0151813, -0.0062230, -0.0070336, 0.0073820
1: -0.0072335, -0.0045957, -0.0072188, -0.0046932, -0.0019830, 0.0020813
2: -0.0148104, 0.0046519, -0.0147023, 0.0039327, -0.0146313, 0.0153560
3: -0.0003326, 0.0022429, -0.0003183, 0.0021477, -0.0019362, 0.0020321
4: 0.0026153, 0.0171602, 0.0031528, 0.0170795, -0.0114761, 0.0109345
5: 0.9962329, 1.0002738, 0.9963822, 1.0002514, -0.0031884, 0.0030379
6: 0.0044642, 0.0081322, 0.0045998, 0.0081119, -0.0028941, 0.0027575
7: -0.0067218, 0.0069665, -0.0062160, 0.0068906, -0.0108003, 0.0102906
8: -0.0146149, -0.0039613, -0.0145558, -0.0043549, -0.0080092, 0.0084059
9: -0.0036680, -0.0027488, -0.0036340, -0.0027539, -0.0007252, 0.0006910

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021271, upper bound: 0.0020576
time: 2.05 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021271, upper bound: 0.0021786
time: 2.07 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0150178, -0.0060266, -0.0147800, -0.0058700, -0.0069440, 0.0065729
1: -0.0071727, -0.0046378, -0.0071057, -0.0045936, -0.0019578, 0.0018532
2: -0.0143622, 0.0043414, -0.0138675, 0.0046671, -0.0144450, 0.0136730
3: -0.0002733, 0.0022018, -0.0002079, 0.0022449, -0.0019116, 0.0018094
4: 0.0028474, 0.0168253, 0.0026040, 0.0164556, -0.0102184, 0.0107953
5: 0.9962974, 1.0001808, 0.9962296, 1.0000781, -0.0028390, 0.0029992
6: 0.0045227, 0.0080478, 0.0044614, 0.0079545, -0.0025769, 0.0027224
7: -0.0065034, 0.0066513, -0.0067325, 0.0063034, -0.0096166, 0.0101595
8: -0.0143696, -0.0041313, -0.0140988, -0.0039529, -0.0079072, 0.0074846
9: -0.0036533, -0.0027700, -0.0036687, -0.0027934, -0.0006457, 0.0006822

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0020461
time: 2.37 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021292
time: 1.57 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0151673, -0.0060016, -0.0152442, -0.0059578, -0.0069735, 0.0069912
1: -0.0072149, -0.0046307, -0.0072366, -0.0046184, -0.0019661, 0.0019711
2: -0.0146732, 0.0043933, -0.0148332, 0.0044844, -0.0145062, 0.0145431
3: -0.0003145, 0.0022087, -0.0003356, 0.0022207, -0.0019197, 0.0019246
4: 0.0028086, 0.0170577, 0.0027405, 0.0171773, -0.0108686, 0.0108411
5: 0.9962866, 1.0002453, 0.9962677, 1.0002786, -0.0030196, 0.0030120
6: 0.0045130, 0.0081064, 0.0044958, 0.0081365, -0.0027409, 0.0027340
7: -0.0065399, 0.0068700, -0.0066040, 0.0069826, -0.0102286, 0.0102026
8: -0.0145398, -0.0041028, -0.0146274, -0.0040529, -0.0079407, 0.0079609
9: -0.0036558, -0.0027553, -0.0036601, -0.0027478, -0.0006868, 0.0006851

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021270, upper bound: 0.0020461
time: 1.56 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021270, upper bound: 0.0021552
time: 1.88 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0150822, -0.0059019, -0.0147938, -0.0058665, -0.0069982, 0.0067559
1: -0.0071909, -0.0046026, -0.0071096, -0.0045927, -0.0019730, 0.0019048
2: -0.0144962, 0.0046006, -0.0138962, 0.0046743, -0.0145576, 0.0140537
3: -0.0002910, 0.0022361, -0.0002116, 0.0022459, -0.0019265, 0.0018598
4: 0.0026537, 0.0169254, 0.0025986, 0.0164770, -0.0105029, 0.0108795
5: 0.9962435, 1.0002086, 0.9962282, 1.0000840, -0.0029180, 0.0030226
6: 0.0044739, 0.0080730, 0.0044600, 0.0079599, -0.0026487, 0.0027436
7: -0.0066858, 0.0067456, -0.0067375, 0.0063236, -0.0098844, 0.0102388
8: -0.0144430, -0.0039893, -0.0141145, -0.0039490, -0.0079689, 0.0076930
9: -0.0036656, -0.0027637, -0.0036690, -0.0027920, -0.0006637, 0.0006875

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020282, upper bound: 0.0020577
time: 1.59 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020282, upper bound: 0.0021520
time: 1.88 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0152332, -0.0058773, -0.0152592, -0.0059540, -0.0070277, 0.0071395
1: -0.0072335, -0.0045957, -0.0072408, -0.0046173, -0.0019814, 0.0020129
2: -0.0148104, 0.0046519, -0.0148645, 0.0044924, -0.0146190, 0.0148516
3: -0.0003326, 0.0022429, -0.0003398, 0.0022218, -0.0019346, 0.0019654
4: 0.0026153, 0.0171602, 0.0027346, 0.0172006, -0.0110991, 0.0109253
5: 0.9962329, 1.0002738, 0.9962659, 1.0002850, -0.0030837, 0.0030354
6: 0.0044642, 0.0081322, 0.0044943, 0.0081424, -0.0027990, 0.0027552
7: -0.0067218, 0.0069665, -0.0066096, 0.0070046, -0.0104455, 0.0102820
8: -0.0146149, -0.0039613, -0.0146445, -0.0040486, -0.0080025, 0.0081298
9: -0.0036680, -0.0027488, -0.0036604, -0.0027463, -0.0007014, 0.0006904

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021270, upper bound: 0.0020576
time: 1.46 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021270, upper bound: 0.0021786
time: 2.16 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.85 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0020142
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0021024
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0021286, upper bound: 0.0020142
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0021286, upper bound: 0.0021285
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0020299
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0021287
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0021287, upper bound: 0.0020299
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0021287, upper bound: 0.0021556
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0020576, upper bound: 0.0020116
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0020576, upper bound: 0.0020997
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0021520, upper bound: 0.0020116
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0021520, upper bound: 0.0021262
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0020576, upper bound: 0.0020281
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0020576, upper bound: 0.0021272
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0021521, upper bound: 0.0020281
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0021521, upper bound: 0.0021536
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0020461
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021292
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0021271, upper bound: 0.0020461
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0021271, upper bound: 0.0021552
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0020282, upper bound: 0.0020576
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0020282, upper bound: 0.0021522
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0021271, upper bound: 0.0020576
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0021271, upper bound: 0.0021786
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0020461
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0020281, upper bound: 0.0021292
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0021270, upper bound: 0.0020461
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0021270, upper bound: 0.0021552
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0020282, upper bound: 0.0020577
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0020282, upper bound: 0.0021520
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0021270, upper bound: 0.0020576
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.85
Output dim: 5, lower bound: -0.0021270, upper bound: 0.0021786

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0145236, -0.0062049, -0.0146761, -0.0061490, -0.0065049, 0.0065228
1: -0.0070334, -0.0046881, -0.0070764, -0.0046723, -0.0018340, 0.0018390
2: -0.0133342, 0.0039704, -0.0136515, 0.0040866, -0.0135315, 0.0135688
3: -0.0001373, 0.0021527, -0.0001793, 0.0021681, -0.0017907, 0.0017956
4: 0.0031246, 0.0160570, 0.0030378, 0.0162941, -0.0101405, 0.0101126
5: 0.9963744, 0.9999673, 0.9963502, 1.0000333, -0.0028173, 0.0028096
6: 0.0045927, 0.0078540, 0.0045707, 0.0079138, -0.0025573, 0.0025502
7: -0.0062425, 0.0059283, -0.0063243, 0.0061515, -0.0095433, 0.0095171
8: -0.0138069, -0.0043343, -0.0139806, -0.0042707, -0.0074072, 0.0074276
9: -0.0036358, -0.0028185, -0.0036413, -0.0028036, -0.0006408, 0.0006391

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020202, upper bound: 0.0020257
time: 1.47 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020202, upper bound: 0.0020257
time: 2.00 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0150048, -0.0062870, -0.0146761, -0.0061490, -0.0069082, 0.0063969
1: -0.0071691, -0.0047112, -0.0070764, -0.0046723, -0.0019477, 0.0018035
2: -0.0143353, 0.0037996, -0.0136515, 0.0040866, -0.0143705, 0.0133068
3: -0.0002697, 0.0021301, -0.0001793, 0.0021681, -0.0019017, 0.0017609
4: 0.0032522, 0.0168052, 0.0030378, 0.0162941, -0.0099447, 0.0107396
5: 0.9964098, 1.0001752, 0.9963502, 1.0000333, -0.0027629, 0.0029838
6: 0.0046248, 0.0080427, 0.0045707, 0.0079138, -0.0025079, 0.0027084
7: -0.0061224, 0.0066324, -0.0063243, 0.0061515, -0.0093590, 0.0101072
8: -0.0143549, -0.0044278, -0.0139806, -0.0042707, -0.0078664, 0.0072842
9: -0.0036277, -0.0027713, -0.0036413, -0.0028036, -0.0006284, 0.0006787

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020202, upper bound: 0.0021024
time: 1.47 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020202, upper bound: 0.0021024
time: 1.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0145236, -0.0062049, -0.0151616, -0.0062268, -0.0063714, 0.0069539
1: -0.0070334, -0.0046881, -0.0072133, -0.0046942, -0.0017963, 0.0019606
2: -0.0133342, 0.0039704, -0.0146613, 0.0039247, -0.0132539, 0.0144655
3: -0.0001373, 0.0021527, -0.0003129, 0.0021467, -0.0017539, 0.0019143
4: 0.0031246, 0.0160570, 0.0031588, 0.0170488, -0.0108106, 0.0099051
5: 0.9963744, 0.9999673, 0.9963838, 1.0002429, -0.0030035, 0.0027519
6: 0.0045927, 0.0078540, 0.0046013, 0.0081041, -0.0027263, 0.0024979
7: -0.0062425, 0.0059283, -0.0062104, 0.0068617, -0.0101740, 0.0093218
8: -0.0138069, -0.0043343, -0.0145333, -0.0043593, -0.0072552, 0.0079184
9: -0.0036358, -0.0028185, -0.0036336, -0.0027559, -0.0006832, 0.0006259

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020203, upper bound: 0.0020142
time: 2.13 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020203, upper bound: 0.0020142
time: 1.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0150048, -0.0062870, -0.0151616, -0.0062268, -0.0067654, 0.0067632
1: -0.0071691, -0.0047112, -0.0072133, -0.0046942, -0.0019074, 0.0019068
2: -0.0143353, 0.0037996, -0.0146613, 0.0039247, -0.0140734, 0.0140689
3: -0.0002697, 0.0021301, -0.0003129, 0.0021467, -0.0018624, 0.0018618
4: 0.0032522, 0.0168052, 0.0031588, 0.0170488, -0.0105142, 0.0105176
5: 0.9964098, 1.0001752, 0.9963838, 1.0002429, -0.0029212, 0.0029221
6: 0.0046248, 0.0080427, 0.0046013, 0.0081041, -0.0026515, 0.0026524
7: -0.0061224, 0.0066324, -0.0062104, 0.0068617, -0.0098950, 0.0098982
8: -0.0143549, -0.0044278, -0.0145333, -0.0043593, -0.0077038, 0.0077013
9: -0.0036277, -0.0027713, -0.0036336, -0.0027559, -0.0006644, 0.0006646

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020203, upper bound: 0.0021286
time: 1.36 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020203, upper bound: 0.0021286
time: 1.43 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0146018, -0.0060723, -0.0146946, -0.0061453, -0.0064781, 0.0066789
1: -0.0070554, -0.0046507, -0.0070816, -0.0046712, -0.0018264, 0.0018830
2: -0.0134968, 0.0042463, -0.0136898, 0.0040944, -0.0134757, 0.0138935
3: -0.0001588, 0.0021892, -0.0001843, 0.0021691, -0.0017833, 0.0018386
4: 0.0029184, 0.0161786, 0.0030320, 0.0163228, -0.0103832, 0.0100709
5: 0.9963171, 1.0000011, 0.9963486, 1.0000412, -0.0028847, 0.0027980
6: 0.0045407, 0.0078847, 0.0045693, 0.0079210, -0.0026185, 0.0025397
7: -0.0064366, 0.0060427, -0.0063297, 0.0061784, -0.0097717, 0.0094778
8: -0.0138959, -0.0041833, -0.0140015, -0.0042665, -0.0073766, 0.0076053
9: -0.0036488, -0.0028109, -0.0036416, -0.0028018, -0.0006562, 0.0006364

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020142, upper bound: 0.0020473
time: 1.91 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020142, upper bound: 0.0020473
time: 2.12 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0150879, -0.0061514, -0.0146946, -0.0061453, -0.0068964, 0.0065757
1: -0.0071925, -0.0046730, -0.0070816, -0.0046712, -0.0019444, 0.0018539
2: -0.0145080, 0.0040818, -0.0136898, 0.0040944, -0.0143460, 0.0136789
3: -0.0002926, 0.0021675, -0.0001843, 0.0021691, -0.0018985, 0.0018102
4: 0.0030414, 0.0169342, 0.0030320, 0.0163228, -0.0102227, 0.0107213
5: 0.9963512, 1.0002111, 0.9963486, 1.0000412, -0.0028402, 0.0029787
6: 0.0045717, 0.0080752, 0.0045693, 0.0079210, -0.0025780, 0.0027038
7: -0.0063208, 0.0067538, -0.0063297, 0.0061784, -0.0096207, 0.0100899
8: -0.0144494, -0.0042734, -0.0140015, -0.0042665, -0.0078530, 0.0074878
9: -0.0036411, -0.0027631, -0.0036416, -0.0028018, -0.0006460, 0.0006775

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020142, upper bound: 0.0021287
time: 2.06 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020142, upper bound: 0.0021287
time: 2.06 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0146018, -0.0060723, -0.0151813, -0.0062230, -0.0063571, 0.0071150
1: -0.0070554, -0.0046507, -0.0072188, -0.0046932, -0.0017923, 0.0020060
2: -0.0134968, 0.0042463, -0.0147023, 0.0039327, -0.0132241, 0.0148006
3: -0.0001588, 0.0021892, -0.0003183, 0.0021477, -0.0017500, 0.0019586
4: 0.0029184, 0.0161786, 0.0031528, 0.0170795, -0.0110610, 0.0098829
5: 0.9963171, 1.0000011, 0.9963822, 1.0002514, -0.0030731, 0.0027458
6: 0.0045407, 0.0078847, 0.0045998, 0.0081119, -0.0027894, 0.0024923
7: -0.0064366, 0.0060427, -0.0062160, 0.0068906, -0.0104096, 0.0093009
8: -0.0138959, -0.0041833, -0.0145558, -0.0043549, -0.0072389, 0.0081018
9: -0.0036488, -0.0028109, -0.0036340, -0.0027539, -0.0006990, 0.0006245

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020142, upper bound: 0.0020299
time: 2.05 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020142, upper bound: 0.0020299
time: 2.00 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0150879, -0.0061514, -0.0151813, -0.0062230, -0.0067195, 0.0069204
1: -0.0071925, -0.0046730, -0.0072188, -0.0046932, -0.0018945, 0.0019511
2: -0.0145080, 0.0040818, -0.0147023, 0.0039327, -0.0139780, 0.0143959
3: -0.0002926, 0.0021675, -0.0003183, 0.0021477, -0.0018498, 0.0019051
4: 0.0030414, 0.0169342, 0.0031528, 0.0170795, -0.0107586, 0.0104463
5: 0.9963512, 1.0002111, 0.9963822, 1.0002514, -0.0029891, 0.0029023
6: 0.0045717, 0.0080752, 0.0045998, 0.0081119, -0.0027132, 0.0026344
7: -0.0063208, 0.0067538, -0.0062160, 0.0068906, -0.0101250, 0.0098311
8: -0.0144494, -0.0042734, -0.0145558, -0.0043549, -0.0076516, 0.0078803
9: -0.0036411, -0.0027631, -0.0036340, -0.0027539, -0.0006799, 0.0006601

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020142, upper bound: 0.0021556
time: 1.96 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020142, upper bound: 0.0021556
time: 1.91 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0145236, -0.0062049, -0.0147800, -0.0058700, -0.0069461, 0.0067644
1: -0.0070334, -0.0046881, -0.0071057, -0.0045936, -0.0019584, 0.0019072
2: -0.0133342, 0.0039704, -0.0138675, 0.0046671, -0.0144492, 0.0140714
3: -0.0001373, 0.0021527, -0.0002079, 0.0022449, -0.0019121, 0.0018621
4: 0.0031246, 0.0160570, 0.0026040, 0.0164556, -0.0105161, 0.0107985
5: 0.9963744, 0.9999673, 0.9962296, 1.0000781, -0.0029217, 0.0030001
6: 0.0045927, 0.0078540, 0.0044614, 0.0079545, -0.0026520, 0.0027232
7: -0.0062425, 0.0059283, -0.0067325, 0.0063034, -0.0098968, 0.0101625
8: -0.0138069, -0.0043343, -0.0140988, -0.0039529, -0.0079095, 0.0077027
9: -0.0036358, -0.0028185, -0.0036687, -0.0027934, -0.0006646, 0.0006824

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020499, upper bound: 0.0020232
time: 1.65 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020499, upper bound: 0.0020232
time: 2.13 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0150048, -0.0062870, -0.0147800, -0.0058700, -0.0073494, 0.0066385
1: -0.0071691, -0.0047112, -0.0071057, -0.0045936, -0.0020721, 0.0018716
2: -0.0143353, 0.0037996, -0.0138675, 0.0046671, -0.0152882, 0.0138094
3: -0.0002697, 0.0021301, -0.0002079, 0.0022449, -0.0020232, 0.0018275
4: 0.0032522, 0.0168052, 0.0026040, 0.0164556, -0.0103203, 0.0114255
5: 0.9964098, 1.0001752, 0.9962296, 1.0000781, -0.0028673, 0.0031743
6: 0.0046248, 0.0080427, 0.0044614, 0.0079545, -0.0026026, 0.0028813
7: -0.0061224, 0.0066324, -0.0067325, 0.0063034, -0.0097126, 0.0107526
8: -0.0143549, -0.0044278, -0.0140988, -0.0039529, -0.0083688, 0.0075593
9: -0.0036277, -0.0027713, -0.0036687, -0.0027934, -0.0006522, 0.0007220

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020499, upper bound: 0.0020996
time: 2.02 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020499, upper bound: 0.0020996
time: 2.23 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0145236, -0.0062049, -0.0152442, -0.0059578, -0.0067732, 0.0071183
1: -0.0070334, -0.0046881, -0.0072366, -0.0046184, -0.0019096, 0.0020069
2: -0.0133342, 0.0039704, -0.0148332, 0.0044844, -0.0140896, 0.0148076
3: -0.0001373, 0.0021527, -0.0003356, 0.0022207, -0.0018645, 0.0019595
4: 0.0031246, 0.0160570, 0.0027405, 0.0171773, -0.0110663, 0.0105297
5: 0.9963744, 0.9999673, 0.9962677, 1.0002786, -0.0030745, 0.0029254
6: 0.0045927, 0.0078540, 0.0044958, 0.0081365, -0.0027907, 0.0026554
7: -0.0062425, 0.0059283, -0.0066040, 0.0069826, -0.0104146, 0.0099096
8: -0.0138069, -0.0043343, -0.0146274, -0.0040529, -0.0077126, 0.0081057
9: -0.0036358, -0.0028185, -0.0036601, -0.0027478, -0.0006993, 0.0006654

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020499, upper bound: 0.0020116
time: 2.13 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020499, upper bound: 0.0020115
time: 2.27 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0150048, -0.0062870, -0.0152442, -0.0059578, -0.0072094, 0.0069889
1: -0.0071691, -0.0047112, -0.0072366, -0.0046184, -0.0020326, 0.0019704
2: -0.0143353, 0.0037996, -0.0148332, 0.0044844, -0.0149970, 0.0145383
3: -0.0002697, 0.0021301, -0.0003356, 0.0022207, -0.0019846, 0.0019239
4: 0.0032522, 0.0168052, 0.0027405, 0.0171773, -0.0108650, 0.0112078
5: 0.9964098, 1.0001752, 0.9962677, 1.0002786, -0.0030186, 0.0031139
6: 0.0046248, 0.0080427, 0.0044958, 0.0081365, -0.0027400, 0.0028265
7: -0.0061224, 0.0066324, -0.0066040, 0.0069826, -0.0102252, 0.0105478
8: -0.0143549, -0.0044278, -0.0146274, -0.0040529, -0.0082094, 0.0079583
9: -0.0036277, -0.0027713, -0.0036601, -0.0027478, -0.0006866, 0.0007083

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020499, upper bound: 0.0021261
time: 1.51 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020499, upper bound: 0.0021262
time: 2.03 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0146018, -0.0060723, -0.0147938, -0.0058665, -0.0069259, 0.0069210
1: -0.0070554, -0.0046507, -0.0071096, -0.0045927, -0.0019527, 0.0019513
2: -0.0134968, 0.0042463, -0.0138962, 0.0046743, -0.0144073, 0.0143972
3: -0.0001588, 0.0021892, -0.0002116, 0.0022459, -0.0019066, 0.0019052
4: 0.0029184, 0.0161786, 0.0025986, 0.0164770, -0.0107595, 0.0107671
5: 0.9963171, 1.0000011, 0.9962282, 1.0000840, -0.0029893, 0.0029914
6: 0.0045407, 0.0078847, 0.0044600, 0.0079599, -0.0027134, 0.0027153
7: -0.0064366, 0.0060427, -0.0067375, 0.0063236, -0.0101259, 0.0101330
8: -0.0138959, -0.0041833, -0.0141145, -0.0039490, -0.0078866, 0.0078810
9: -0.0036488, -0.0028109, -0.0036690, -0.0027920, -0.0006799, 0.0006804

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020462, upper bound: 0.0020450
time: 1.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020461, upper bound: 0.0020450
time: 1.58 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0150879, -0.0061514, -0.0147938, -0.0058665, -0.0073443, 0.0068179
1: -0.0071925, -0.0046730, -0.0071096, -0.0045927, -0.0020706, 0.0019222
2: -0.0145080, 0.0040818, -0.0138962, 0.0046743, -0.0152776, 0.0141825
3: -0.0002926, 0.0021675, -0.0002116, 0.0022459, -0.0020217, 0.0018768
4: 0.0030414, 0.0169342, 0.0025986, 0.0164770, -0.0105991, 0.0114175
5: 0.9963512, 1.0002111, 0.9962282, 1.0000840, -0.0029448, 0.0031721
6: 0.0045717, 0.0080752, 0.0044600, 0.0079599, -0.0026729, 0.0028793
7: -0.0063208, 0.0067538, -0.0067375, 0.0063236, -0.0099750, 0.0107451
8: -0.0144494, -0.0042734, -0.0141145, -0.0039490, -0.0083630, 0.0077635
9: -0.0036411, -0.0027631, -0.0036690, -0.0027920, -0.0006698, 0.0007215

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020462, upper bound: 0.0021272
time: 1.56 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020461, upper bound: 0.0021272
time: 1.55 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0146018, -0.0060723, -0.0152592, -0.0059540, -0.0067628, 0.0072794
1: -0.0070554, -0.0046507, -0.0072408, -0.0046173, -0.0019067, 0.0020523
2: -0.0134968, 0.0042463, -0.0148645, 0.0044924, -0.0140680, 0.0151426
3: -0.0001588, 0.0021892, -0.0003398, 0.0022218, -0.0018617, 0.0020039
4: 0.0029184, 0.0161786, 0.0027346, 0.0172006, -0.0113166, 0.0105136
5: 0.9963171, 1.0000011, 0.9962659, 1.0002850, -0.0031441, 0.0029210
6: 0.0045407, 0.0078847, 0.0044943, 0.0081424, -0.0028539, 0.0026514
7: -0.0064366, 0.0060427, -0.0066096, 0.0070046, -0.0106502, 0.0098944
8: -0.0138959, -0.0041833, -0.0146445, -0.0040486, -0.0077009, 0.0082891
9: -0.0036488, -0.0028109, -0.0036604, -0.0027463, -0.0007151, 0.0006644

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020462, upper bound: 0.0020281
time: 2.15 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020462, upper bound: 0.0020281
time: 2.11 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0150879, -0.0061514, -0.0152592, -0.0059540, -0.0071709, 0.0071480
1: -0.0071925, -0.0046730, -0.0072408, -0.0046173, -0.0020217, 0.0020153
2: -0.0145080, 0.0040818, -0.0148645, 0.0044924, -0.0149169, 0.0148693
3: -0.0002926, 0.0021675, -0.0003398, 0.0022218, -0.0019740, 0.0019677
4: 0.0030414, 0.0169342, 0.0027346, 0.0172006, -0.0111124, 0.0111480
5: 0.9963512, 1.0002111, 0.9962659, 1.0002850, -0.0030873, 0.0030972
6: 0.0045717, 0.0080752, 0.0044943, 0.0081424, -0.0028024, 0.0028114
7: -0.0063208, 0.0067538, -0.0066096, 0.0070046, -0.0104580, 0.0104915
8: -0.0144494, -0.0042734, -0.0146445, -0.0040486, -0.0081655, 0.0081395
9: -0.0036411, -0.0027631, -0.0036604, -0.0027463, -0.0007022, 0.0007045

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020462, upper bound: 0.0021536
time: 2.13 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020462, upper bound: 0.0021536
time: 2.09 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0146279, -0.0059244, -0.0146761, -0.0061490, -0.0066545, 0.0069726
1: -0.0070628, -0.0046090, -0.0070764, -0.0046723, -0.0018761, 0.0019658
2: -0.0135512, 0.0045538, -0.0136515, 0.0040866, -0.0138426, 0.0145044
3: -0.0001660, 0.0022299, -0.0001793, 0.0021681, -0.0018319, 0.0019194
4: 0.0026886, 0.0162192, 0.0030378, 0.0162941, -0.0108397, 0.0103451
5: 0.9962532, 1.0000124, 0.9963502, 1.0000333, -0.0030116, 0.0028742
6: 0.0044827, 0.0078949, 0.0045707, 0.0079138, -0.0027336, 0.0026089
7: -0.0066529, 0.0060810, -0.0063243, 0.0061515, -0.0102013, 0.0097359
8: -0.0139257, -0.0040149, -0.0139806, -0.0042707, -0.0075775, 0.0079397
9: -0.0036633, -0.0028083, -0.0036413, -0.0028036, -0.0006850, 0.0006537

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020182, upper bound: 0.0020558
time: 1.66 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020182, upper bound: 0.0020558
time: 2.23 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0150919, -0.0060165, -0.0146761, -0.0061490, -0.0070066, 0.0068051
1: -0.0071936, -0.0046349, -0.0070764, -0.0046723, -0.0019754, 0.0019186
2: -0.0145165, 0.0043623, -0.0136515, 0.0040866, -0.0145751, 0.0141559
3: -0.0002937, 0.0022046, -0.0001793, 0.0021681, -0.0019288, 0.0018733
4: 0.0028318, 0.0169406, 0.0030378, 0.0162941, -0.0105792, 0.0108925
5: 0.9962931, 1.0002128, 0.9963502, 1.0000333, -0.0029392, 0.0030263
6: 0.0045188, 0.0080768, 0.0045707, 0.0079138, -0.0026679, 0.0027469
7: -0.0065182, 0.0067598, -0.0063243, 0.0061515, -0.0099562, 0.0102510
8: -0.0144540, -0.0041198, -0.0139806, -0.0042707, -0.0079784, 0.0077490
9: -0.0036543, -0.0027627, -0.0036413, -0.0028036, -0.0006685, 0.0006883

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020182, upper bound: 0.0021292
time: 1.54 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020182, upper bound: 0.0021292
time: 2.09 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0146279, -0.0059244, -0.0151616, -0.0062268, -0.0065333, 0.0074037
1: -0.0070628, -0.0046090, -0.0072133, -0.0046942, -0.0018420, 0.0020874
2: -0.0135512, 0.0045538, -0.0146613, 0.0039247, -0.0135906, 0.0154011
3: -0.0001660, 0.0022299, -0.0003129, 0.0021467, -0.0017985, 0.0020381
4: 0.0026886, 0.0162192, 0.0031588, 0.0170488, -0.0115098, 0.0101568
5: 0.9962532, 1.0000124, 0.9963838, 1.0002429, -0.0031978, 0.0028219
6: 0.0044827, 0.0078949, 0.0046013, 0.0081041, -0.0029026, 0.0025614
7: -0.0066529, 0.0060810, -0.0062104, 0.0068617, -0.0108320, 0.0095587
8: -0.0139257, -0.0040149, -0.0145333, -0.0043593, -0.0074395, 0.0084306
9: -0.0036633, -0.0028083, -0.0036336, -0.0027559, -0.0007274, 0.0006418

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0020462
time: 2.23 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0020462
time: 1.78 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0150919, -0.0060165, -0.0151616, -0.0062268, -0.0068802, 0.0072169
1: -0.0071936, -0.0046349, -0.0072133, -0.0046942, -0.0019398, 0.0020347
2: -0.0145165, 0.0043623, -0.0146613, 0.0039247, -0.0143121, 0.0150126
3: -0.0002937, 0.0022046, -0.0003129, 0.0021467, -0.0018940, 0.0019867
4: 0.0028318, 0.0169406, 0.0031588, 0.0170488, -0.0112195, 0.0106960
5: 0.9962931, 1.0002128, 0.9963838, 1.0002429, -0.0031171, 0.0029717
6: 0.0045188, 0.0080768, 0.0046013, 0.0081041, -0.0028294, 0.0026974
7: -0.0065182, 0.0067598, -0.0062104, 0.0068617, -0.0105588, 0.0100661
8: -0.0144540, -0.0041198, -0.0145333, -0.0043593, -0.0078345, 0.0082179
9: -0.0036543, -0.0027627, -0.0036336, -0.0027559, -0.0007090, 0.0006759

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0021551
time: 2.08 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0021552
time: 2.06 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0146912, -0.0058057, -0.0146946, -0.0061453, -0.0067120, 0.0071153
1: -0.0070806, -0.0045755, -0.0070816, -0.0046712, -0.0018924, 0.0020061
2: -0.0136827, 0.0048008, -0.0136898, 0.0040944, -0.0139622, 0.0148014
3: -0.0001834, 0.0022626, -0.0001843, 0.0021691, -0.0018477, 0.0019587
4: 0.0025041, 0.0163175, 0.0030320, 0.0163228, -0.0110616, 0.0104345
5: 0.9962019, 1.0000397, 0.9963486, 1.0000412, -0.0030732, 0.0028990
6: 0.0044362, 0.0079197, 0.0045693, 0.0079210, -0.0027896, 0.0026314
7: -0.0068265, 0.0061734, -0.0063297, 0.0061784, -0.0104102, 0.0098200
8: -0.0139977, -0.0038798, -0.0140015, -0.0042665, -0.0076430, 0.0081023
9: -0.0036750, -0.0028021, -0.0036416, -0.0028018, -0.0006990, 0.0006594

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0020728
time: 1.60 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0020729
time: 2.11 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0151599, -0.0058933, -0.0146946, -0.0061453, -0.0070641, 0.0069637
1: -0.0072128, -0.0046002, -0.0070816, -0.0046712, -0.0019916, 0.0019633
2: -0.0146578, 0.0046185, -0.0136898, 0.0040944, -0.0146949, 0.0144859
3: -0.0003124, 0.0022385, -0.0001843, 0.0021691, -0.0019446, 0.0019170
4: 0.0026403, 0.0170462, 0.0030320, 0.0163228, -0.0108259, 0.0109820
5: 0.9962398, 1.0002421, 0.9963486, 1.0000412, -0.0030077, 0.0030511
6: 0.0044705, 0.0081035, 0.0045693, 0.0079210, -0.0027301, 0.0027695
7: -0.0066983, 0.0068592, -0.0063297, 0.0061784, -0.0101884, 0.0103353
8: -0.0145314, -0.0039795, -0.0140015, -0.0042665, -0.0080440, 0.0079296
9: -0.0036664, -0.0027560, -0.0036416, -0.0028018, -0.0006841, 0.0006940

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0021521
time: 1.56 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020115, upper bound: 0.0021522
time: 2.14 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0146912, -0.0058057, -0.0151813, -0.0062230, -0.0065910, 0.0075514
1: -0.0070806, -0.0045755, -0.0072188, -0.0046932, -0.0018583, 0.0021290
2: -0.0136827, 0.0048008, -0.0147023, 0.0039327, -0.0137107, 0.0157084
3: -0.0001834, 0.0022626, -0.0003183, 0.0021477, -0.0018144, 0.0020788
4: 0.0025041, 0.0163175, 0.0031528, 0.0170795, -0.0117395, 0.0102465
5: 0.9962019, 1.0000397, 0.9963822, 1.0002514, -0.0032616, 0.0028468
6: 0.0044362, 0.0079197, 0.0045998, 0.0081119, -0.0029605, 0.0025840
7: -0.0068265, 0.0061734, -0.0062160, 0.0068906, -0.0110482, 0.0096431
8: -0.0139977, -0.0038798, -0.0145558, -0.0043549, -0.0075052, 0.0085988
9: -0.0036750, -0.0028021, -0.0036340, -0.0027539, -0.0007419, 0.0006475

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0020577
time: 1.62 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0020576
time: 2.08 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0151599, -0.0058933, -0.0151813, -0.0062230, -0.0069392, 0.0073547
1: -0.0072128, -0.0046002, -0.0072188, -0.0046932, -0.0019564, 0.0020736
2: -0.0146578, 0.0046185, -0.0147023, 0.0039327, -0.0144350, 0.0152992
3: -0.0003124, 0.0022385, -0.0003183, 0.0021477, -0.0019102, 0.0020246
4: 0.0026403, 0.0170462, 0.0031528, 0.0170795, -0.0114337, 0.0107878
5: 0.9962398, 1.0002421, 0.9963822, 1.0002514, -0.0031766, 0.0029972
6: 0.0044705, 0.0081035, 0.0045998, 0.0081119, -0.0028834, 0.0027205
7: -0.0066983, 0.0068592, -0.0062160, 0.0068906, -0.0107604, 0.0101525
8: -0.0145314, -0.0039795, -0.0145558, -0.0043549, -0.0079017, 0.0083748
9: -0.0036664, -0.0027560, -0.0036340, -0.0027539, -0.0007225, 0.0006817

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0021786
time: 2.02 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0021787
time: 1.48 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0146279, -0.0059244, -0.0147800, -0.0058700, -0.0066120, 0.0067172
1: -0.0070628, -0.0046090, -0.0071057, -0.0045936, -0.0018642, 0.0018938
2: -0.0135512, 0.0045538, -0.0138675, 0.0046671, -0.0137544, 0.0139732
3: -0.0001660, 0.0022299, -0.0002079, 0.0022449, -0.0018202, 0.0018491
4: 0.0026886, 0.0162192, 0.0026040, 0.0164556, -0.0104427, 0.0102792
5: 0.9962532, 1.0000124, 0.9962296, 1.0000781, -0.0029013, 0.0028559
6: 0.0044827, 0.0078949, 0.0044614, 0.0079545, -0.0026335, 0.0025923
7: -0.0066529, 0.0060810, -0.0067325, 0.0063034, -0.0098277, 0.0096738
8: -0.0139257, -0.0040149, -0.0140988, -0.0039529, -0.0075292, 0.0076489
9: -0.0036633, -0.0028083, -0.0036687, -0.0027934, -0.0006599, 0.0006496

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020182, upper bound: 0.0020558
time: 2.18 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0020559
time: 2.06 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0150919, -0.0060165, -0.0147800, -0.0058700, -0.0070347, 0.0065766
1: -0.0071936, -0.0046349, -0.0071057, -0.0045936, -0.0019834, 0.0018542
2: -0.0145165, 0.0043623, -0.0138675, 0.0046671, -0.0146337, 0.0136807
3: -0.0002937, 0.0022046, -0.0002079, 0.0022449, -0.0019365, 0.0018104
4: 0.0028318, 0.0169406, 0.0026040, 0.0164556, -0.0102241, 0.0109363
5: 0.9962931, 1.0002128, 0.9962296, 1.0000781, -0.0028406, 0.0030384
6: 0.0045188, 0.0080768, 0.0044614, 0.0079545, -0.0025784, 0.0027580
7: -0.0065182, 0.0067598, -0.0067325, 0.0063034, -0.0096220, 0.0102923
8: -0.0144540, -0.0041198, -0.0140988, -0.0039529, -0.0080105, 0.0074888
9: -0.0036543, -0.0027627, -0.0036687, -0.0027934, -0.0006461, 0.0006911

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0021292
time: 1.56 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0021291
time: 1.50 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0146279, -0.0059244, -0.0152442, -0.0059578, -0.0064758, 0.0071487
1: -0.0070628, -0.0046090, -0.0072366, -0.0046184, -0.0018258, 0.0020155
2: -0.0135512, 0.0045538, -0.0148332, 0.0044844, -0.0134709, 0.0148708
3: -0.0001660, 0.0022299, -0.0003356, 0.0022207, -0.0017827, 0.0019679
4: 0.0026886, 0.0162192, 0.0027405, 0.0171773, -0.0111135, 0.0100673
5: 0.9962532, 1.0000124, 0.9962677, 1.0002786, -0.0030877, 0.0027970
6: 0.0044827, 0.0078949, 0.0044958, 0.0081365, -0.0028027, 0.0025388
7: -0.0066529, 0.0060810, -0.0066040, 0.0069826, -0.0104591, 0.0094745
8: -0.0139257, -0.0040149, -0.0146274, -0.0040529, -0.0073740, 0.0081403
9: -0.0036633, -0.0028083, -0.0036601, -0.0027478, -0.0007023, 0.0006362

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0020461
time: 1.64 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0020461
time: 2.09 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0150919, -0.0060165, -0.0152442, -0.0059578, -0.0068607, 0.0069666
1: -0.0071936, -0.0046349, -0.0072366, -0.0046184, -0.0019343, 0.0019641
2: -0.0145165, 0.0043623, -0.0148332, 0.0044844, -0.0142716, 0.0144919
3: -0.0002937, 0.0022046, -0.0003356, 0.0022207, -0.0018886, 0.0019178
4: 0.0028318, 0.0169406, 0.0027405, 0.0171773, -0.0108303, 0.0106657
5: 0.9962931, 1.0002128, 0.9962677, 1.0002786, -0.0030090, 0.0029632
6: 0.0045188, 0.0080768, 0.0044958, 0.0081365, -0.0027312, 0.0026897
7: -0.0065182, 0.0067598, -0.0066040, 0.0069826, -0.0101925, 0.0100376
8: -0.0144540, -0.0041198, -0.0146274, -0.0040529, -0.0078123, 0.0079329
9: -0.0036543, -0.0027627, -0.0036601, -0.0027478, -0.0006844, 0.0006740

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0021551
time: 1.95 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0020461
time: 1.51 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0146912, -0.0058057, -0.0147938, -0.0058665, -0.0066665, 0.0068661
1: -0.0070806, -0.0045755, -0.0071096, -0.0045927, -0.0018795, 0.0019358
2: -0.0136827, 0.0048008, -0.0138962, 0.0046743, -0.0138676, 0.0142830
3: -0.0001834, 0.0022626, -0.0002116, 0.0022459, -0.0018352, 0.0018901
4: 0.0025041, 0.0163175, 0.0025986, 0.0164770, -0.0106742, 0.0103638
5: 0.9962019, 1.0000397, 0.9962282, 1.0000840, -0.0029656, 0.0028794
6: 0.0044362, 0.0079197, 0.0044600, 0.0079599, -0.0026919, 0.0026136
7: -0.0068265, 0.0061734, -0.0067375, 0.0063236, -0.0100456, 0.0097535
8: -0.0139977, -0.0038798, -0.0141145, -0.0039490, -0.0075912, 0.0078185
9: -0.0036750, -0.0028021, -0.0036690, -0.0027920, -0.0006745, 0.0006549

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0020728
time: 1.62 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0020729
time: 1.99 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0151599, -0.0058933, -0.0147938, -0.0058665, -0.0070889, 0.0067589
1: -0.0072128, -0.0046002, -0.0071096, -0.0045927, -0.0019986, 0.0019056
2: -0.0146578, 0.0046185, -0.0138962, 0.0046743, -0.0147464, 0.0140598
3: -0.0003124, 0.0022385, -0.0002116, 0.0022459, -0.0019514, 0.0018606
4: 0.0026403, 0.0170462, 0.0025986, 0.0164770, -0.0105074, 0.0110205
5: 0.9962398, 1.0002421, 0.9962282, 1.0000840, -0.0029193, 0.0030618
6: 0.0044705, 0.0081035, 0.0044600, 0.0079599, -0.0026498, 0.0027792
7: -0.0066983, 0.0068592, -0.0067375, 0.0063236, -0.0098887, 0.0103715
8: -0.0145314, -0.0039795, -0.0141145, -0.0039490, -0.0080722, 0.0076964
9: -0.0036664, -0.0027560, -0.0036690, -0.0027920, -0.0006640, 0.0006964

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0021521
time: 1.71 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0021521
time: 1.62 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0146912, -0.0058057, -0.0152592, -0.0059540, -0.0065305, 0.0072989
1: -0.0070806, -0.0045755, -0.0072408, -0.0046173, -0.0018412, 0.0020578
2: -0.0136827, 0.0048008, -0.0148645, 0.0044924, -0.0135848, 0.0151832
3: -0.0001834, 0.0022626, -0.0003398, 0.0022218, -0.0017977, 0.0020093
4: 0.0025041, 0.0163175, 0.0027346, 0.0172006, -0.0113470, 0.0101524
5: 0.9962019, 1.0000397, 0.9962659, 1.0002850, -0.0031525, 0.0028206
6: 0.0044362, 0.0079197, 0.0044943, 0.0081424, -0.0028615, 0.0025603
7: -0.0068265, 0.0061734, -0.0066096, 0.0070046, -0.0106787, 0.0095546
8: -0.0139977, -0.0038798, -0.0146445, -0.0040486, -0.0074363, 0.0083113
9: -0.0036750, -0.0028021, -0.0036604, -0.0027463, -0.0007171, 0.0006416

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0020577
time: 1.62 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020115, upper bound: 0.0020577
time: 2.08 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0151599, -0.0058933, -0.0152592, -0.0059540, -0.0069117, 0.0071122
1: -0.0072128, -0.0046002, -0.0072408, -0.0046173, -0.0019487, 0.0020052
2: -0.0146578, 0.0046185, -0.0148645, 0.0044924, -0.0143778, 0.0147949
3: -0.0003124, 0.0022385, -0.0003398, 0.0022218, -0.0019027, 0.0019579
4: 0.0026403, 0.0170462, 0.0027346, 0.0172006, -0.0110568, 0.0107451
5: 0.9962398, 1.0002421, 0.9962659, 1.0002850, -0.0030719, 0.0029853
6: 0.0044705, 0.0081035, 0.0044943, 0.0081424, -0.0027884, 0.0027098
7: -0.0066983, 0.0068592, -0.0066096, 0.0070046, -0.0104057, 0.0101123
8: -0.0145314, -0.0039795, -0.0146445, -0.0040486, -0.0078704, 0.0080987
9: -0.0036664, -0.0027560, -0.0036604, -0.0027463, -0.0006987, 0.0006790

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0021787
time: 1.97 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020115, upper bound: 0.0021786
time: 1.48 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.86 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020202, upper bound: 0.0020257
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020202, upper bound: 0.0020257
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020202, upper bound: 0.0021024
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020202, upper bound: 0.0021024
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020203, upper bound: 0.0020142
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020203, upper bound: 0.0020142
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020203, upper bound: 0.0021286
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020203, upper bound: 0.0021286
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020142, upper bound: 0.0020473
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020142, upper bound: 0.0020473
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020142, upper bound: 0.0021287
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020142, upper bound: 0.0021287
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020142, upper bound: 0.0020299
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020142, upper bound: 0.0020299
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020142, upper bound: 0.0021556
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020142, upper bound: 0.0021556
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020499, upper bound: 0.0020232
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020499, upper bound: 0.0020232
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020499, upper bound: 0.0020996
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020499, upper bound: 0.0020996
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020499, upper bound: 0.0020116
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020499, upper bound: 0.0020115
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020499, upper bound: 0.0021261
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020499, upper bound: 0.0021262
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020462, upper bound: 0.0020450
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020461, upper bound: 0.0020450
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020462, upper bound: 0.0021272
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020461, upper bound: 0.0021272
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020462, upper bound: 0.0020281
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020462, upper bound: 0.0020281
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020462, upper bound: 0.0021536
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020462, upper bound: 0.0021536
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020182, upper bound: 0.0020558
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020182, upper bound: 0.0020558
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020182, upper bound: 0.0021292
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020182, upper bound: 0.0021292
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0020462
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0020462
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0021551
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0021552
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0020728
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0020729
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0021521
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020115, upper bound: 0.0021522
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0020577
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0020576
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0021786
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0021787
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020182, upper bound: 0.0020558
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0020559
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0021292
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0021291
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0020461
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0020461
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0021551
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020184, upper bound: 0.0020461
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0020728
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0020729
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0021521
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0021521
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0020577
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020115, upper bound: 0.0020577
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020117, upper bound: 0.0021787
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 5, lower bound: -0.0020115, upper bound: 0.0021786

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0145236, -0.0062049, -0.0145239, -0.0062049, -0.0064576, 0.0063708
1: -0.0070334, -0.0046881, -0.0070335, -0.0046881, -0.0018206, 0.0017962
2: -0.0133342, 0.0039704, -0.0133348, 0.0039704, -0.0134330, 0.0132525
3: -0.0001373, 0.0021527, -0.0001374, 0.0021527, -0.0017776, 0.0017538
4: 0.0031246, 0.0160570, 0.0031246, 0.0160575, -0.0099041, 0.0100390
5: 0.9963744, 0.9999673, 0.9963744, 0.9999675, -0.0027516, 0.0027891
6: 0.0045927, 0.0078540, 0.0045927, 0.0078541, -0.0024977, 0.0025317
7: -0.0062425, 0.0059283, -0.0062425, 0.0059287, -0.0093208, 0.0094478
8: -0.0138069, -0.0043343, -0.0138072, -0.0043343, -0.0073533, 0.0072544
9: -0.0036358, -0.0028185, -0.0036358, -0.0028185, -0.0006259, 0.0006344

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019291, upper bound: 0.0019185
time: 1.42 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019136, upper bound: 0.0019104
time: 1.77 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0145236, -0.0062049, -0.0146018, -0.0060723, -0.0065932, 0.0064467
1: -0.0070334, -0.0046881, -0.0070554, -0.0046507, -0.0018589, 0.0018176
2: -0.0133342, 0.0039704, -0.0134968, 0.0042463, -0.0137152, 0.0134105
3: -0.0001373, 0.0021527, -0.0001588, 0.0021892, -0.0018150, 0.0017747
4: 0.0031246, 0.0160570, 0.0029184, 0.0161786, -0.0100222, 0.0102499
5: 0.9963744, 0.9999673, 0.9963171, 1.0000011, -0.0027845, 0.0028477
6: 0.0045927, 0.0078540, 0.0045407, 0.0078847, -0.0025275, 0.0025849
7: -0.0062425, 0.0059283, -0.0064366, 0.0060427, -0.0094320, 0.0096463
8: -0.0138069, -0.0043343, -0.0138959, -0.0041833, -0.0075077, 0.0073409
9: -0.0036358, -0.0028185, -0.0036488, -0.0028109, -0.0006333, 0.0006477

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019291, upper bound: 0.0019185
time: 1.83 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019136, upper bound: 0.0019103
time: 2.04 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0150048, -0.0062870, -0.0145239, -0.0062049, -0.0068609, 0.0062448
1: -0.0071691, -0.0047112, -0.0070335, -0.0046881, -0.0019343, 0.0017606
2: -0.0143353, 0.0037996, -0.0133348, 0.0039704, -0.0142720, 0.0129905
3: -0.0002697, 0.0021301, -0.0001374, 0.0021527, -0.0018887, 0.0017191
4: 0.0032522, 0.0168052, 0.0031246, 0.0160575, -0.0097083, 0.0106660
5: 0.9964098, 1.0001752, 0.9963744, 0.9999675, -0.0026972, 0.0029633
6: 0.0046248, 0.0080427, 0.0045927, 0.0078541, -0.0024483, 0.0026898
7: -0.0061224, 0.0066324, -0.0062425, 0.0059287, -0.0091366, 0.0100379
8: -0.0143549, -0.0044278, -0.0138072, -0.0043343, -0.0078125, 0.0071110
9: -0.0036277, -0.0027713, -0.0036358, -0.0028185, -0.0006135, 0.0006740

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019254, upper bound: 0.0020199
time: 1.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019136, upper bound: 0.0020246
time: 1.78 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0150048, -0.0062870, -0.0146018, -0.0060723, -0.0069965, 0.0063208
1: -0.0071691, -0.0047112, -0.0070554, -0.0046507, -0.0019726, 0.0017821
2: -0.0143353, 0.0037996, -0.0134968, 0.0042463, -0.0145542, 0.0131485
3: -0.0002697, 0.0021301, -0.0001588, 0.0021892, -0.0019260, 0.0017400
4: 0.0032522, 0.0168052, 0.0029184, 0.0161786, -0.0098264, 0.0108769
5: 0.9964098, 1.0001752, 0.9963171, 1.0000011, -0.0027301, 0.0030219
6: 0.0046248, 0.0080427, 0.0045407, 0.0078847, -0.0024781, 0.0027430
7: -0.0061224, 0.0066324, -0.0064366, 0.0060427, -0.0092477, 0.0102364
8: -0.0143549, -0.0044278, -0.0138959, -0.0041833, -0.0079670, 0.0071975
9: -0.0036277, -0.0027713, -0.0036488, -0.0028109, -0.0006210, 0.0006874

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019254, upper bound: 0.0020199
time: 1.79 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019136, upper bound: 0.0020244
time: 1.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0145236, -0.0062049, -0.0150048, -0.0062870, -0.0063195, 0.0068609
1: -0.0070334, -0.0046881, -0.0071691, -0.0047112, -0.0017817, 0.0019343
2: -0.0133342, 0.0039704, -0.0143353, 0.0037996, -0.0131458, 0.0142720
3: -0.0001373, 0.0021527, -0.0002697, 0.0021301, -0.0017396, 0.0018887
4: 0.0031246, 0.0160570, 0.0032522, 0.0168052, -0.0106660, 0.0098243
5: 0.9963744, 0.9999673, 0.9964098, 1.0001752, -0.0029633, 0.0027295
6: 0.0045927, 0.0078540, 0.0046248, 0.0080427, -0.0026898, 0.0024775
7: -0.0062425, 0.0059283, -0.0061224, 0.0066324, -0.0100379, 0.0092458
8: -0.0138069, -0.0043343, -0.0143549, -0.0044278, -0.0071960, 0.0078125
9: -0.0036358, -0.0028185, -0.0036277, -0.0027713, -0.0006740, 0.0006208

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020460, upper bound: 0.0019184
time: 1.50 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020340, upper bound: 0.0019102
time: 1.44 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0145236, -0.0062049, -0.0150879, -0.0061514, -0.0064833, 0.0068810
1: -0.0070334, -0.0046881, -0.0071925, -0.0046730, -0.0018279, 0.0019400
2: -0.0133342, 0.0039704, -0.0145080, 0.0040818, -0.0134866, 0.0143139
3: -0.0001373, 0.0021527, -0.0002926, 0.0021675, -0.0017847, 0.0018942
4: 0.0031246, 0.0160570, 0.0030414, 0.0169342, -0.0106973, 0.0100790
5: 0.9963744, 0.9999673, 0.9963512, 1.0002111, -0.0029720, 0.0028002
6: 0.0045927, 0.0078540, 0.0045717, 0.0080752, -0.0026977, 0.0025418
7: -0.0062425, 0.0059283, -0.0063208, 0.0067538, -0.0100673, 0.0094855
8: -0.0138069, -0.0043343, -0.0144494, -0.0042734, -0.0073826, 0.0078354
9: -0.0036358, -0.0028185, -0.0036411, -0.0027631, -0.0006760, 0.0006369

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020460, upper bound: 0.0019183
time: 1.47 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020340, upper bound: 0.0019102
time: 1.93 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0150048, -0.0062870, -0.0150048, -0.0062870, -0.0067167, 0.0067167
1: -0.0071691, -0.0047112, -0.0071691, -0.0047112, -0.0018937, 0.0018937
2: -0.0143353, 0.0037996, -0.0143353, 0.0037996, -0.0139721, 0.0139721
3: -0.0002697, 0.0021301, -0.0002697, 0.0021301, -0.0018490, 0.0018490
4: 0.0032522, 0.0168052, 0.0032522, 0.0168052, -0.0104419, 0.0104419
5: 0.9964098, 1.0001752, 0.9964098, 1.0001752, -0.0029011, 0.0029011
6: 0.0046248, 0.0080427, 0.0046248, 0.0080427, -0.0026333, 0.0026333
7: -0.0061224, 0.0066324, -0.0061224, 0.0066324, -0.0098269, 0.0098269
8: -0.0143549, -0.0044278, -0.0143549, -0.0044278, -0.0076483, 0.0076483
9: -0.0036277, -0.0027713, -0.0036277, -0.0027713, -0.0006599, 0.0006599

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020094, upper bound: 0.0020670
time: 1.58 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020095, upper bound: 0.0020786
time: 1.40 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0150048, -0.0062870, -0.0150879, -0.0061514, -0.0068535, 0.0066804
1: -0.0071691, -0.0047112, -0.0071925, -0.0046730, -0.0019322, 0.0018834
2: -0.0143353, 0.0037996, -0.0145080, 0.0040818, -0.0142566, 0.0138965
3: -0.0002697, 0.0021301, -0.0002926, 0.0021675, -0.0018866, 0.0018390
4: 0.0032522, 0.0168052, 0.0030414, 0.0169342, -0.0103854, 0.0106545
5: 0.9964098, 1.0001752, 0.9963512, 1.0002111, -0.0028854, 0.0029601
6: 0.0046248, 0.0080427, 0.0045717, 0.0080752, -0.0026190, 0.0026869
7: -0.0061224, 0.0066324, -0.0063208, 0.0067538, -0.0097738, 0.0100271
8: -0.0143549, -0.0044278, -0.0144494, -0.0042734, -0.0078041, 0.0076070
9: -0.0036277, -0.0027713, -0.0036411, -0.0027631, -0.0006563, 0.0006733

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020094, upper bound: 0.0020670
time: 1.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020095, upper bound: 0.0020786
time: 4.87 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0146018, -0.0060723, -0.0145239, -0.0062049, -0.0064467, 0.0065106
1: -0.0070554, -0.0046507, -0.0070335, -0.0046881, -0.0018176, 0.0018356
2: -0.0134968, 0.0042463, -0.0133348, 0.0039704, -0.0134105, 0.0135433
3: -0.0001588, 0.0021892, -0.0001374, 0.0021527, -0.0017747, 0.0017922
4: 0.0029184, 0.0161786, 0.0031246, 0.0160575, -0.0101214, 0.0100222
5: 0.9963171, 1.0000011, 0.9963744, 0.9999675, -0.0028120, 0.0027845
6: 0.0045407, 0.0078847, 0.0045927, 0.0078541, -0.0025525, 0.0025275
7: -0.0064366, 0.0060427, -0.0062425, 0.0059287, -0.0095254, 0.0094320
8: -0.0138959, -0.0041833, -0.0138072, -0.0043343, -0.0073409, 0.0074136
9: -0.0036488, -0.0028109, -0.0036358, -0.0028185, -0.0006396, 0.0006333

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019228, upper bound: 0.0019343
time: 1.49 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019103, upper bound: 0.0019259
time: 1.90 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0146018, -0.0060723, -0.0146018, -0.0060723, -0.0064877, 0.0064877
1: -0.0070554, -0.0046507, -0.0070554, -0.0046507, -0.0018291, 0.0018291
2: -0.0134968, 0.0042463, -0.0134968, 0.0042463, -0.0134957, 0.0134957
3: -0.0001588, 0.0021892, -0.0001588, 0.0021892, -0.0017859, 0.0017859
4: 0.0029184, 0.0161786, 0.0029184, 0.0161786, -0.0100858, 0.0100858
5: 0.9963171, 1.0000011, 0.9963171, 1.0000011, -0.0028021, 0.0028021
6: 0.0045407, 0.0078847, 0.0045407, 0.0078847, -0.0025435, 0.0025435
7: -0.0064366, 0.0060427, -0.0064366, 0.0060427, -0.0094919, 0.0094919
8: -0.0138959, -0.0041833, -0.0138959, -0.0041833, -0.0073876, 0.0073876
9: -0.0036488, -0.0028109, -0.0036488, -0.0028109, -0.0006374, 0.0006374

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019228, upper bound: 0.0019340
time: 1.83 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019104, upper bound: 0.0019261
time: 1.93 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0150879, -0.0061514, -0.0145239, -0.0062049, -0.0068810, 0.0064074
1: -0.0071925, -0.0046730, -0.0070335, -0.0046881, -0.0019400, 0.0018065
2: -0.0145080, 0.0040818, -0.0133348, 0.0039704, -0.0143139, 0.0133286
3: -0.0002926, 0.0021675, -0.0001374, 0.0021527, -0.0018942, 0.0017638
4: 0.0030414, 0.0169342, 0.0031246, 0.0160575, -0.0099610, 0.0106973
5: 0.9963512, 1.0002111, 0.9963744, 0.9999675, -0.0027675, 0.0029720
6: 0.0045717, 0.0080752, 0.0045927, 0.0078541, -0.0025120, 0.0026977
7: -0.0063208, 0.0067538, -0.0062425, 0.0059287, -0.0093744, 0.0100673
8: -0.0144494, -0.0042734, -0.0138072, -0.0043343, -0.0078354, 0.0072961
9: -0.0036411, -0.0027631, -0.0036358, -0.0028185, -0.0006295, 0.0006760

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019202, upper bound: 0.0020453
time: 1.83 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019104, upper bound: 0.0020544
time: 1.75 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0150879, -0.0061514, -0.0146018, -0.0060723, -0.0069060, 0.0063635
1: -0.0071925, -0.0046730, -0.0070554, -0.0046507, -0.0019471, 0.0017941
2: -0.0145080, 0.0040818, -0.0134968, 0.0042463, -0.0143660, 0.0132373
3: -0.0002926, 0.0021675, -0.0001588, 0.0021892, -0.0019011, 0.0017517
4: 0.0030414, 0.0169342, 0.0029184, 0.0161786, -0.0098928, 0.0107362
5: 0.9963512, 1.0002111, 0.9963171, 1.0000011, -0.0027485, 0.0029828
6: 0.0045717, 0.0080752, 0.0045407, 0.0078847, -0.0024948, 0.0027075
7: -0.0063208, 0.0067538, -0.0064366, 0.0060427, -0.0093102, 0.0101040
8: -0.0144494, -0.0042734, -0.0138959, -0.0041833, -0.0078640, 0.0072461
9: -0.0036411, -0.0027631, -0.0036488, -0.0028109, -0.0006252, 0.0006785

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019202, upper bound: 0.0020456
time: 1.80 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019104, upper bound: 0.0020544
time: 1.85 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0146018, -0.0060723, -0.0150048, -0.0062870, -0.0063208, 0.0069965
1: -0.0070554, -0.0046507, -0.0071691, -0.0047112, -0.0017821, 0.0019726
2: -0.0134968, 0.0042463, -0.0143353, 0.0037996, -0.0131485, 0.0145542
3: -0.0001588, 0.0021892, -0.0002697, 0.0021301, -0.0017400, 0.0019260
4: 0.0029184, 0.0161786, 0.0032522, 0.0168052, -0.0108769, 0.0098264
5: 0.9963171, 1.0000011, 0.9964098, 1.0001752, -0.0030219, 0.0027301
6: 0.0045407, 0.0078847, 0.0046248, 0.0080427, -0.0027430, 0.0024781
7: -0.0064366, 0.0060427, -0.0061224, 0.0066324, -0.0102364, 0.0092477
8: -0.0138959, -0.0041833, -0.0143549, -0.0044278, -0.0071975, 0.0079670
9: -0.0036488, -0.0028109, -0.0036277, -0.0027713, -0.0006874, 0.0006210

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020345, upper bound: 0.0019341
time: 1.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020246, upper bound: 0.0019261
time: 2.02 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0146018, -0.0060723, -0.0150879, -0.0061514, -0.0063635, 0.0069060
1: -0.0070554, -0.0046507, -0.0071925, -0.0046730, -0.0017941, 0.0019471
2: -0.0134968, 0.0042463, -0.0145080, 0.0040818, -0.0132373, 0.0143660
3: -0.0001588, 0.0021892, -0.0002926, 0.0021675, -0.0017517, 0.0019011
4: 0.0029184, 0.0161786, 0.0030414, 0.0169342, -0.0107362, 0.0098928
5: 0.9963171, 1.0000011, 0.9963512, 1.0002111, -0.0029828, 0.0027485
6: 0.0045407, 0.0078847, 0.0045717, 0.0080752, -0.0027075, 0.0024948
7: -0.0064366, 0.0060427, -0.0063208, 0.0067538, -0.0101040, 0.0093102
8: -0.0138959, -0.0041833, -0.0144494, -0.0042734, -0.0072461, 0.0078640
9: -0.0036488, -0.0028109, -0.0036411, -0.0027631, -0.0006785, 0.0006252

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020345, upper bound: 0.0019339
time: 2.10 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020246, upper bound: 0.0019259
time: 2.15 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0150879, -0.0061514, -0.0150048, -0.0062870, -0.0066804, 0.0068535
1: -0.0071925, -0.0046730, -0.0071691, -0.0047112, -0.0018834, 0.0019322
2: -0.0145080, 0.0040818, -0.0143353, 0.0037996, -0.0138965, 0.0142566
3: -0.0002926, 0.0021675, -0.0002697, 0.0021301, -0.0018390, 0.0018866
4: 0.0030414, 0.0169342, 0.0032522, 0.0168052, -0.0106545, 0.0103854
5: 0.9963512, 1.0002111, 0.9964098, 1.0001752, -0.0029601, 0.0028854
6: 0.0045717, 0.0080752, 0.0046248, 0.0080427, -0.0026869, 0.0026190
7: -0.0063208, 0.0067538, -0.0061224, 0.0066324, -0.0100271, 0.0097738
8: -0.0144494, -0.0042734, -0.0143549, -0.0044278, -0.0076070, 0.0078041
9: -0.0036411, -0.0027631, -0.0036277, -0.0027713, -0.0006733, 0.0006563

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020042, upper bound: 0.0020908
time: 1.47 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020042, upper bound: 0.0021063
time: 1.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0150879, -0.0061514, -0.0150879, -0.0061514, -0.0067248, 0.0067248
1: -0.0071925, -0.0046730, -0.0071925, -0.0046730, -0.0018960, 0.0018960
2: -0.0145080, 0.0040818, -0.0145080, 0.0040818, -0.0139890, 0.0139890
3: -0.0002926, 0.0021675, -0.0002926, 0.0021675, -0.0018512, 0.0018512
4: 0.0030414, 0.0169342, 0.0030414, 0.0169342, -0.0104545, 0.0104545
5: 0.9963512, 1.0002111, 0.9963512, 1.0002111, -0.0029046, 0.0029046
6: 0.0045717, 0.0080752, 0.0045717, 0.0080752, -0.0026365, 0.0026365
7: -0.0063208, 0.0067538, -0.0063208, 0.0067538, -0.0098388, 0.0098388
8: -0.0144494, -0.0042734, -0.0144494, -0.0042734, -0.0076576, 0.0076576
9: -0.0036411, -0.0027631, -0.0036411, -0.0027631, -0.0006607, 0.0006607

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020042, upper bound: 0.0020908
time: 1.51 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020042, upper bound: 0.0021062
time: 1.73 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0145236, -0.0062049, -0.0146279, -0.0059244, -0.0069005, 0.0066068
1: -0.0070334, -0.0046881, -0.0070628, -0.0046090, -0.0019455, 0.0018627
2: -0.0133342, 0.0039704, -0.0135512, 0.0045538, -0.0143545, 0.0137434
3: -0.0001373, 0.0021527, -0.0001660, 0.0022299, -0.0018996, 0.0018187
4: 0.0031246, 0.0160570, 0.0026886, 0.0162192, -0.0102710, 0.0107277
5: 0.9963744, 0.9999673, 0.9962532, 1.0000124, -0.0028536, 0.0029805
6: 0.0045927, 0.0078540, 0.0044827, 0.0078949, -0.0025902, 0.0027054
7: -0.0062425, 0.0059283, -0.0066529, 0.0060810, -0.0096661, 0.0100959
8: -0.0138069, -0.0043343, -0.0139257, -0.0040149, -0.0078577, 0.0075232
9: -0.0036358, -0.0028185, -0.0036633, -0.0028083, -0.0006491, 0.0006779

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019544, upper bound: 0.0019152
time: 1.84 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019378, upper bound: 0.0019074
time: 1.44 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0145236, -0.0062049, -0.0146912, -0.0058057, -0.0070246, 0.0066783
1: -0.0070334, -0.0046881, -0.0070806, -0.0045755, -0.0019805, 0.0018829
2: -0.0133342, 0.0039704, -0.0136827, 0.0048008, -0.0146127, 0.0138921
3: -0.0001373, 0.0021527, -0.0001834, 0.0022626, -0.0019338, 0.0018384
4: 0.0031246, 0.0160570, 0.0025041, 0.0163175, -0.0103821, 0.0109206
5: 0.9963744, 0.9999673, 0.9962019, 1.0000397, -0.0028845, 0.0030341
6: 0.0045927, 0.0078540, 0.0044362, 0.0079197, -0.0026182, 0.0027540
7: -0.0062425, 0.0059283, -0.0068265, 0.0061734, -0.0097707, 0.0102775
8: -0.0138069, -0.0043343, -0.0139977, -0.0038798, -0.0079990, 0.0076046
9: -0.0036358, -0.0028185, -0.0036750, -0.0028021, -0.0006561, 0.0006901

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019544, upper bound: 0.0019152
time: 1.94 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019378, upper bound: 0.0019074
time: 1.97 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0150048, -0.0062870, -0.0146279, -0.0059244, -0.0073038, 0.0064808
1: -0.0071691, -0.0047112, -0.0070628, -0.0046090, -0.0020592, 0.0018272
2: -0.0143353, 0.0037996, -0.0135512, 0.0045538, -0.0151935, 0.0134814
3: -0.0002697, 0.0021301, -0.0001660, 0.0022299, -0.0020106, 0.0017840
4: 0.0032522, 0.0168052, 0.0026886, 0.0162192, -0.0100752, 0.0113547
5: 0.9964098, 1.0001752, 0.9962532, 1.0000124, -0.0027992, 0.0031547
6: 0.0046248, 0.0080427, 0.0044827, 0.0078949, -0.0025408, 0.0028635
7: -0.0061224, 0.0066324, -0.0066529, 0.0060810, -0.0094818, 0.0106860
8: -0.0143549, -0.0044278, -0.0139257, -0.0040149, -0.0083169, 0.0073797
9: -0.0036277, -0.0027713, -0.0036633, -0.0028083, -0.0006367, 0.0007175

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019510, upper bound: 0.0020156
time: 1.48 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019378, upper bound: 0.0020215
time: 1.71 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0150048, -0.0062870, -0.0146912, -0.0058057, -0.0074280, 0.0065523
1: -0.0071691, -0.0047112, -0.0070806, -0.0045755, -0.0020942, 0.0018473
2: -0.0143353, 0.0037996, -0.0136827, 0.0048008, -0.0154517, 0.0136302
3: -0.0002697, 0.0021301, -0.0001834, 0.0022626, -0.0020448, 0.0018037
4: 0.0032522, 0.0168052, 0.0025041, 0.0163175, -0.0101863, 0.0115476
5: 0.9964098, 1.0001752, 0.9962019, 1.0000397, -0.0028301, 0.0032083
6: 0.0046248, 0.0080427, 0.0044362, 0.0079197, -0.0025688, 0.0029121
7: -0.0061224, 0.0066324, -0.0068265, 0.0061734, -0.0095865, 0.0108676
8: -0.0143549, -0.0044278, -0.0139977, -0.0038798, -0.0084583, 0.0074612
9: -0.0036277, -0.0027713, -0.0036750, -0.0028021, -0.0006437, 0.0007297

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019510, upper bound: 0.0020156
time: 1.86 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019378, upper bound: 0.0020215
time: 1.93 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0145236, -0.0062049, -0.0150919, -0.0060165, -0.0067234, 0.0069588
1: -0.0070334, -0.0046881, -0.0071936, -0.0046349, -0.0018956, 0.0019620
2: -0.0133342, 0.0039704, -0.0145165, 0.0043623, -0.0139860, 0.0144758
3: -0.0001373, 0.0021527, -0.0002937, 0.0022046, -0.0018508, 0.0019156
4: 0.0031246, 0.0160570, 0.0028318, 0.0169406, -0.0108183, 0.0104522
5: 0.9963744, 0.9999673, 0.9962931, 1.0002128, -0.0030057, 0.0029039
6: 0.0045927, 0.0078540, 0.0045188, 0.0080768, -0.0027282, 0.0026359
7: -0.0062425, 0.0059283, -0.0065182, 0.0067598, -0.0101813, 0.0098367
8: -0.0138069, -0.0043343, -0.0144540, -0.0041198, -0.0076559, 0.0079241
9: -0.0036358, -0.0028185, -0.0036543, -0.0027627, -0.0006837, 0.0006605

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020668, upper bound: 0.0019154
time: 1.53 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020561, upper bound: 0.0019074
time: 1.58 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0145236, -0.0062049, -0.0151599, -0.0058933, -0.0068628, 0.0070455
1: -0.0070334, -0.0046881, -0.0072128, -0.0046002, -0.0019349, 0.0019864
2: -0.0133342, 0.0039704, -0.0146578, 0.0046185, -0.0142761, 0.0146560
3: -0.0001373, 0.0021527, -0.0003124, 0.0022385, -0.0018892, 0.0019395
4: 0.0031246, 0.0160570, 0.0026403, 0.0170462, -0.0109530, 0.0106691
5: 0.9963744, 0.9999673, 0.9962398, 1.0002421, -0.0030431, 0.0029642
6: 0.0045927, 0.0078540, 0.0044705, 0.0081035, -0.0027622, 0.0026906
7: -0.0062425, 0.0059283, -0.0066983, 0.0068592, -0.0103080, 0.0100408
8: -0.0138069, -0.0043343, -0.0145314, -0.0039795, -0.0078148, 0.0080227
9: -0.0036358, -0.0028185, -0.0036664, -0.0027560, -0.0006922, 0.0006742

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020668, upper bound: 0.0019154
time: 1.96 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020561, upper bound: 0.0019074
time: 1.99 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0150048, -0.0062870, -0.0150919, -0.0060165, -0.0071626, 0.0068309
1: -0.0071691, -0.0047112, -0.0071936, -0.0046349, -0.0020194, 0.0019259
2: -0.0143353, 0.0037996, -0.0145165, 0.0043623, -0.0148997, 0.0142097
3: -0.0002697, 0.0021301, -0.0002937, 0.0022046, -0.0019717, 0.0018804
4: 0.0032522, 0.0168052, 0.0028318, 0.0169406, -0.0106194, 0.0111351
5: 0.9964098, 1.0001752, 0.9962931, 1.0002128, -0.0029504, 0.0030937
6: 0.0046248, 0.0080427, 0.0045188, 0.0080768, -0.0026781, 0.0028081
7: -0.0061224, 0.0066324, -0.0065182, 0.0067598, -0.0099941, 0.0104794
8: -0.0143549, -0.0044278, -0.0144540, -0.0041198, -0.0081561, 0.0077784
9: -0.0036277, -0.0027713, -0.0036543, -0.0027627, -0.0006711, 0.0007037

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020395, upper bound: 0.0020650
time: 1.97 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020394, upper bound: 0.0020757
time: 2.08 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0150048, -0.0062870, -0.0151599, -0.0058933, -0.0072853, 0.0068994
1: -0.0071691, -0.0047112, -0.0072128, -0.0046002, -0.0020540, 0.0019452
2: -0.0143353, 0.0037996, -0.0146578, 0.0046185, -0.0151549, 0.0143522
3: -0.0002697, 0.0021301, -0.0003124, 0.0022385, -0.0020055, 0.0018993
4: 0.0032522, 0.0168052, 0.0026403, 0.0170462, -0.0107259, 0.0113258
5: 0.9964098, 1.0001752, 0.9962398, 1.0002421, -0.0029800, 0.0031467
6: 0.0046248, 0.0080427, 0.0044705, 0.0081035, -0.0027049, 0.0028562
7: -0.0061224, 0.0066324, -0.0066983, 0.0068592, -0.0100943, 0.0106589
8: -0.0143549, -0.0044278, -0.0145314, -0.0039795, -0.0082958, 0.0078564
9: -0.0036277, -0.0027713, -0.0036664, -0.0027560, -0.0006778, 0.0007157

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020395, upper bound: 0.0020650
time: 2.05 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020394, upper bound: 0.0020758
time: 2.13 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0146018, -0.0060723, -0.0146279, -0.0059244, -0.0068965, 0.0067466
1: -0.0070554, -0.0046507, -0.0070628, -0.0046090, -0.0019444, 0.0019021
2: -0.0134968, 0.0042463, -0.0135512, 0.0045538, -0.0143461, 0.0140342
3: -0.0001588, 0.0021892, -0.0001660, 0.0022299, -0.0018985, 0.0018572
4: 0.0029184, 0.0161786, 0.0026886, 0.0162192, -0.0104883, 0.0107214
5: 0.9963171, 1.0000011, 0.9962532, 1.0000124, -0.0029140, 0.0029787
6: 0.0045407, 0.0078847, 0.0044827, 0.0078949, -0.0026450, 0.0027038
7: -0.0064366, 0.0060427, -0.0066529, 0.0060810, -0.0098706, 0.0100900
8: -0.0138959, -0.0041833, -0.0139257, -0.0040149, -0.0078531, 0.0076823
9: -0.0036488, -0.0028109, -0.0036633, -0.0028083, -0.0006628, 0.0006775

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019487, upper bound: 0.0019310
time: 1.96 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019345, upper bound: 0.0019230
time: 1.48 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0146018, -0.0060723, -0.0146912, -0.0058057, -0.0069344, 0.0067216
1: -0.0070554, -0.0046507, -0.0070806, -0.0045755, -0.0019551, 0.0018951
2: -0.0134968, 0.0042463, -0.0136827, 0.0048008, -0.0144249, 0.0139823
3: -0.0001588, 0.0021892, -0.0001834, 0.0022626, -0.0019089, 0.0018503
4: 0.0029184, 0.0161786, 0.0025041, 0.0163175, -0.0104495, 0.0107803
5: 0.9963171, 1.0000011, 0.9962019, 1.0000397, -0.0029032, 0.0029951
6: 0.0045407, 0.0078847, 0.0044362, 0.0079197, -0.0026352, 0.0027186
7: -0.0064366, 0.0060427, -0.0068265, 0.0061734, -0.0098341, 0.0101454
8: -0.0138959, -0.0041833, -0.0139977, -0.0038798, -0.0078962, 0.0076539
9: -0.0036488, -0.0028109, -0.0036750, -0.0028021, -0.0006603, 0.0006812

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019487, upper bound: 0.0019311
time: 2.01 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019345, upper bound: 0.0019230
time: 1.99 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0150879, -0.0061514, -0.0146279, -0.0059244, -0.0073308, 0.0066434
1: -0.0071925, -0.0046730, -0.0070628, -0.0046090, -0.0020668, 0.0018730
2: -0.0145080, 0.0040818, -0.0135512, 0.0045538, -0.0152495, 0.0138196
3: -0.0002926, 0.0021675, -0.0001660, 0.0022299, -0.0020180, 0.0018288
4: 0.0030414, 0.0169342, 0.0026886, 0.0162192, -0.0103279, 0.0113965
5: 0.9963512, 1.0002111, 0.9962532, 1.0000124, -0.0028694, 0.0031663
6: 0.0045717, 0.0080752, 0.0044827, 0.0078949, -0.0026045, 0.0028740
7: -0.0063208, 0.0067538, -0.0066529, 0.0060810, -0.0097197, 0.0107254
8: -0.0144494, -0.0042734, -0.0139257, -0.0040149, -0.0083476, 0.0075648
9: -0.0036411, -0.0027631, -0.0036633, -0.0028083, -0.0006527, 0.0007202

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019466, upper bound: 0.0020421
time: 1.80 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019345, upper bound: 0.0020514
time: 1.85 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0150879, -0.0061514, -0.0146912, -0.0058057, -0.0073527, 0.0065974
1: -0.0071925, -0.0046730, -0.0070806, -0.0045755, -0.0020730, 0.0018600
2: -0.0145080, 0.0040818, -0.0136827, 0.0048008, -0.0152952, 0.0137239
3: -0.0002926, 0.0021675, -0.0001834, 0.0022626, -0.0020241, 0.0018161
4: 0.0030414, 0.0169342, 0.0025041, 0.0163175, -0.0102564, 0.0114307
5: 0.9963512, 1.0002111, 0.9962019, 1.0000397, -0.0028495, 0.0031758
6: 0.0045717, 0.0080752, 0.0044362, 0.0079197, -0.0025865, 0.0028826
7: -0.0063208, 0.0067538, -0.0068265, 0.0061734, -0.0096524, 0.0107575
8: -0.0144494, -0.0042734, -0.0139977, -0.0038798, -0.0083726, 0.0075125
9: -0.0036411, -0.0027631, -0.0036750, -0.0028021, -0.0006481, 0.0007223

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019466, upper bound: 0.0020421
time: 1.93 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019345, upper bound: 0.0020513
time: 2.03 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0146018, -0.0060723, -0.0150919, -0.0060165, -0.0067290, 0.0070986
1: -0.0070554, -0.0046507, -0.0071936, -0.0046349, -0.0018971, 0.0020014
2: -0.0134968, 0.0042463, -0.0145165, 0.0043623, -0.0139976, 0.0147666
3: -0.0001588, 0.0021892, -0.0002937, 0.0022046, -0.0018524, 0.0019541
4: 0.0029184, 0.0161786, 0.0028318, 0.0169406, -0.0110357, 0.0104610
5: 0.9963171, 1.0000011, 0.9962931, 1.0002128, -0.0030660, 0.0029064
6: 0.0045407, 0.0078847, 0.0045188, 0.0080768, -0.0027830, 0.0026381
7: -0.0064366, 0.0060427, -0.0065182, 0.0067598, -0.0103858, 0.0098449
8: -0.0138959, -0.0041833, -0.0144540, -0.0041198, -0.0076623, 0.0080833
9: -0.0036488, -0.0028109, -0.0036543, -0.0027627, -0.0006974, 0.0006611

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020580, upper bound: 0.0019310
time: 1.61 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020497, upper bound: 0.0019230
time: 1.50 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0146018, -0.0060723, -0.0151599, -0.0058933, -0.0067661, 0.0070738
1: -0.0070554, -0.0046507, -0.0072128, -0.0046002, -0.0019076, 0.0019944
2: -0.0134968, 0.0042463, -0.0146578, 0.0046185, -0.0140749, 0.0147149
3: -0.0001588, 0.0021892, -0.0003124, 0.0022385, -0.0018626, 0.0019473
4: 0.0029184, 0.0161786, 0.0026403, 0.0170462, -0.0109970, 0.0105187
5: 0.9963171, 1.0000011, 0.9962398, 1.0002421, -0.0030553, 0.0029224
6: 0.0045407, 0.0078847, 0.0044705, 0.0081035, -0.0027733, 0.0026527
7: -0.0064366, 0.0060427, -0.0066983, 0.0068592, -0.0103494, 0.0098993
8: -0.0138959, -0.0041833, -0.0145314, -0.0039795, -0.0077046, 0.0080549
9: -0.0036488, -0.0028109, -0.0036664, -0.0027560, -0.0006949, 0.0006647

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020580, upper bound: 0.0019310
time: 2.05 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020497, upper bound: 0.0019229
time: 2.00 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0150879, -0.0061514, -0.0150919, -0.0060165, -0.0071341, 0.0069732
1: -0.0071925, -0.0046730, -0.0071936, -0.0046349, -0.0020114, 0.0019660
2: -0.0145080, 0.0040818, -0.0145165, 0.0043623, -0.0148403, 0.0145057
3: -0.0002926, 0.0021675, -0.0002937, 0.0022046, -0.0019639, 0.0019196
4: 0.0030414, 0.0169342, 0.0028318, 0.0169406, -0.0108406, 0.0110907
5: 0.9963512, 1.0002111, 0.9962931, 1.0002128, -0.0030119, 0.0030813
6: 0.0045717, 0.0080752, 0.0045188, 0.0080768, -0.0027339, 0.0027969
7: -0.0063208, 0.0067538, -0.0065182, 0.0067598, -0.0102023, 0.0104376
8: -0.0144494, -0.0042734, -0.0144540, -0.0041198, -0.0081236, 0.0079404
9: -0.0036411, -0.0027631, -0.0036543, -0.0027627, -0.0006851, 0.0007009

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020360, upper bound: 0.0020889
time: 2.36 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020362, upper bound: 0.0021041
time: 1.92 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0150879, -0.0061514, -0.0151599, -0.0058933, -0.0071746, 0.0069445
1: -0.0071925, -0.0046730, -0.0072128, -0.0046002, -0.0020228, 0.0019579
2: -0.0145080, 0.0040818, -0.0146578, 0.0046185, -0.0149247, 0.0144459
3: -0.0002926, 0.0021675, -0.0003124, 0.0022385, -0.0019750, 0.0019117
4: 0.0030414, 0.0169342, 0.0026403, 0.0170462, -0.0107960, 0.0111538
5: 0.9963512, 1.0002111, 0.9962398, 1.0002421, -0.0029994, 0.0030989
6: 0.0045717, 0.0080752, 0.0044705, 0.0081035, -0.0027226, 0.0028128
7: -0.0063208, 0.0067538, -0.0066983, 0.0068592, -0.0101602, 0.0104969
8: -0.0144494, -0.0042734, -0.0145314, -0.0039795, -0.0081698, 0.0079077
9: -0.0036411, -0.0027631, -0.0036664, -0.0027560, -0.0006822, 0.0007048

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020360, upper bound: 0.0020889
time: 1.85 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020362, upper bound: 0.0021041
time: 2.07 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0146279, -0.0059244, -0.0145239, -0.0062049, -0.0066068, 0.0068205
1: -0.0070628, -0.0046090, -0.0070335, -0.0046881, -0.0018627, 0.0019230
2: -0.0135512, 0.0045538, -0.0133348, 0.0039704, -0.0137434, 0.0141881
3: -0.0001660, 0.0022299, -0.0001374, 0.0021527, -0.0018187, 0.0018776
4: 0.0026886, 0.0162192, 0.0031246, 0.0160575, -0.0106033, 0.0102710
5: 0.9962532, 1.0000124, 0.9963744, 0.9999675, -0.0029459, 0.0028536
6: 0.0044827, 0.0078949, 0.0045927, 0.0078541, -0.0026740, 0.0025902
7: -0.0066529, 0.0060810, -0.0062425, 0.0059287, -0.0099789, 0.0096661
8: -0.0139257, -0.0040149, -0.0138072, -0.0043343, -0.0075232, 0.0077666
9: -0.0036633, -0.0028083, -0.0036358, -0.0028185, -0.0006701, 0.0006491

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019277, upper bound: 0.0019463
time: 1.83 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019109, upper bound: 0.0019344
time: 1.97 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0146279, -0.0059244, -0.0146018, -0.0060723, -0.0067466, 0.0068965
1: -0.0070628, -0.0046090, -0.0070554, -0.0046507, -0.0019021, 0.0019444
2: -0.0135512, 0.0045538, -0.0134968, 0.0042463, -0.0140342, 0.0143461
3: -0.0001660, 0.0022299, -0.0001588, 0.0021892, -0.0018572, 0.0018985
4: 0.0026886, 0.0162192, 0.0029184, 0.0161786, -0.0107214, 0.0104883
5: 0.9962532, 1.0000124, 0.9963171, 1.0000011, -0.0029787, 0.0029140
6: 0.0044827, 0.0078949, 0.0045407, 0.0078847, -0.0027038, 0.0026450
7: -0.0066529, 0.0060810, -0.0064366, 0.0060427, -0.0100900, 0.0098706
8: -0.0139257, -0.0040149, -0.0138959, -0.0041833, -0.0076823, 0.0078531
9: -0.0036633, -0.0028083, -0.0036488, -0.0028109, -0.0006775, 0.0006628

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019277, upper bound: 0.0019465
time: 1.95 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019109, upper bound: 0.0019346
time: 2.09 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0150919, -0.0060165, -0.0145239, -0.0062049, -0.0069588, 0.0066530
1: -0.0071936, -0.0046349, -0.0070335, -0.0046881, -0.0019620, 0.0018757
2: -0.0145165, 0.0043623, -0.0133348, 0.0039704, -0.0144758, 0.0138396
3: -0.0002937, 0.0022046, -0.0001374, 0.0021527, -0.0019156, 0.0018314
4: 0.0028318, 0.0169406, 0.0031246, 0.0160575, -0.0103428, 0.0108183
5: 0.9962931, 1.0002128, 0.9963744, 0.9999675, -0.0028735, 0.0030057
6: 0.0045188, 0.0080768, 0.0045927, 0.0078541, -0.0026083, 0.0027282
7: -0.0065182, 0.0067598, -0.0062425, 0.0059287, -0.0097338, 0.0101813
8: -0.0144540, -0.0041198, -0.0138072, -0.0043343, -0.0079241, 0.0075758
9: -0.0036543, -0.0027627, -0.0036358, -0.0028185, -0.0006536, 0.0006837

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019236, upper bound: 0.0020452
time: 1.51 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019109, upper bound: 0.0020497
time: 1.71 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0150919, -0.0060165, -0.0146018, -0.0060723, -0.0070986, 0.0067290
1: -0.0071936, -0.0046349, -0.0070554, -0.0046507, -0.0020014, 0.0018971
2: -0.0145165, 0.0043623, -0.0134968, 0.0042463, -0.0147666, 0.0139976
3: -0.0002937, 0.0022046, -0.0001588, 0.0021892, -0.0019541, 0.0018524
4: 0.0028318, 0.0169406, 0.0029184, 0.0161786, -0.0104610, 0.0110357
5: 0.9962931, 1.0002128, 0.9963171, 1.0000011, -0.0029064, 0.0030660
6: 0.0045188, 0.0080768, 0.0045407, 0.0078847, -0.0026381, 0.0027830
7: -0.0065182, 0.0067598, -0.0064366, 0.0060427, -0.0098449, 0.0103858
8: -0.0144540, -0.0041198, -0.0138959, -0.0041833, -0.0080833, 0.0076623
9: -0.0036543, -0.0027627, -0.0036488, -0.0028109, -0.0006611, 0.0006974

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019236, upper bound: 0.0020453
time: 2.04 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019109, upper bound: 0.0020495
time: 2.11 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0146279, -0.0059244, -0.0150048, -0.0062870, -0.0064808, 0.0073038
1: -0.0070628, -0.0046090, -0.0071691, -0.0047112, -0.0018272, 0.0020592
2: -0.0135512, 0.0045538, -0.0143353, 0.0037996, -0.0134814, 0.0151935
3: -0.0001660, 0.0022299, -0.0002697, 0.0021301, -0.0017840, 0.0020106
4: 0.0026886, 0.0162192, 0.0032522, 0.0168052, -0.0113547, 0.0100752
5: 0.9962532, 1.0000124, 0.9964098, 1.0001752, -0.0031547, 0.0027992
6: 0.0044827, 0.0078949, 0.0046248, 0.0080427, -0.0028635, 0.0025408
7: -0.0066529, 0.0060810, -0.0061224, 0.0066324, -0.0106860, 0.0094818
8: -0.0139257, -0.0040149, -0.0143549, -0.0044278, -0.0073797, 0.0083169
9: -0.0036633, -0.0028083, -0.0036277, -0.0027713, -0.0007175, 0.0006367

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020440, upper bound: 0.0019464
time: 1.99 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020314, upper bound: 0.0019344
time: 1.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0146279, -0.0059244, -0.0150879, -0.0061514, -0.0066434, 0.0073308
1: -0.0070628, -0.0046090, -0.0071925, -0.0046730, -0.0018730, 0.0020668
2: -0.0135512, 0.0045538, -0.0145080, 0.0040818, -0.0138196, 0.0152495
3: -0.0001660, 0.0022299, -0.0002926, 0.0021675, -0.0018288, 0.0020180
4: 0.0026886, 0.0162192, 0.0030414, 0.0169342, -0.0113965, 0.0103279
5: 0.9962532, 1.0000124, 0.9963512, 1.0002111, -0.0031663, 0.0028694
6: 0.0044827, 0.0078949, 0.0045717, 0.0080752, -0.0028740, 0.0026045
7: -0.0066529, 0.0060810, -0.0063208, 0.0067538, -0.0107254, 0.0097197
8: -0.0139257, -0.0040149, -0.0144494, -0.0042734, -0.0075648, 0.0083476
9: -0.0036633, -0.0028083, -0.0036411, -0.0027631, -0.0007202, 0.0006527

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020440, upper bound: 0.0019463
time: 2.09 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020314, upper bound: 0.0019345
time: 1.71 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0150919, -0.0060165, -0.0150048, -0.0062870, -0.0068309, 0.0071626
1: -0.0071936, -0.0046349, -0.0071691, -0.0047112, -0.0019259, 0.0020194
2: -0.0145165, 0.0043623, -0.0143353, 0.0037996, -0.0142097, 0.0148997
3: -0.0002937, 0.0022046, -0.0002697, 0.0021301, -0.0018804, 0.0019717
4: 0.0028318, 0.0169406, 0.0032522, 0.0168052, -0.0111351, 0.0106194
5: 0.9962931, 1.0002128, 0.9964098, 1.0001752, -0.0030937, 0.0029504
6: 0.0045188, 0.0080768, 0.0046248, 0.0080427, -0.0028081, 0.0026781
7: -0.0065182, 0.0067598, -0.0061224, 0.0066324, -0.0104794, 0.0099941
8: -0.0144540, -0.0041198, -0.0143549, -0.0044278, -0.0077784, 0.0081561
9: -0.0036543, -0.0027627, -0.0036277, -0.0027713, -0.0007037, 0.0006711

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020073, upper bound: 0.0020956
time: 1.59 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020073, upper bound: 0.0021037
time: 1.59 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.98 + 598.15 = 603.14 seconds
