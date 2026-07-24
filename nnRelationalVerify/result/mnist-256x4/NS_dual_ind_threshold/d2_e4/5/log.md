## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.029087099999999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0074664, 0.0208159, -0.0074664, 0.0208159, -0.0282824, 0.0282824)
1: (-0.0114187, 0.0159950, -0.0114187, 0.0159950, -0.0274137, 0.0274137)
2: (-0.0526190, 0.0148103, -0.0526190, 0.0148103, -0.0674294, 0.0674294)
3: (-0.0272276, 0.0312855, -0.0272276, 0.0312855, -0.0585130, 0.0585130)
4: (0.0078912, 0.0179667, 0.0078912, 0.0179667, -0.0100755, 0.0100755)
5: (-0.0395915, 0.0445230, -0.0395915, 0.0445230, -0.0841145, 0.0841145)
6: (0.9707767, 1.0273691, 0.9707767, 1.0273691, -0.0565925, 0.0565925)
7: (0.0009016, 0.0308351, 0.0009016, 0.0308351, -0.0293946, 0.0293946)
8: (-0.0081205, 0.0168595, -0.0081205, 0.0168595, -0.0249800, 0.0249800)
9: (-0.0501618, 0.0064838, -0.0501618, 0.0064838, -0.0566456, 0.0566456)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.85 + 2.99 = 4.84 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0306180, upper bound: 0.0306180

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 189

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0296622, upper bound: 0.0295214
time: 3.01 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0295873, upper bound: 0.0295873
time: 2.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 5.64 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 5.64
Output dim: 6, lower bound: -0.0296622, upper bound: 0.0295214
NS_A2, status: Status.UNKNOWN, split count: 1, time: 5.64
Output dim: 6, lower bound: -0.0295873, upper bound: 0.0295873

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0068348, 0.0207356, -0.0074664, 0.0208159, -0.0276507, 0.0282021
1: -0.0106604, 0.0159185, -0.0114187, 0.0159950, -0.0266554, 0.0273372
2: -0.0511131, 0.0146188, -0.0526190, 0.0148103, -0.0659234, 0.0672379
3: -0.0270614, 0.0299787, -0.0272276, 0.0312855, -0.0583469, 0.0572062
4: 0.0079413, 0.0179526, 0.0078912, 0.0179667, -0.0100254, 0.0100613
5: -0.0393576, 0.0421123, -0.0395915, 0.0445230, -0.0838806, 0.0817038
6: 0.9709343, 1.0257722, 0.9707767, 1.0273691, -0.0564348, 0.0549955
7: 0.0009923, 0.0295581, 0.0009016, 0.0308351, -0.0293154, 0.0282897
8: -0.0075626, 0.0167886, -0.0081205, 0.0168595, -0.0244221, 0.0249090
9: -0.0488967, 0.0063230, -0.0501618, 0.0064838, -0.0553805, 0.0564848

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 189

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0295170, upper bound: 0.0295170
time: 2.12 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0295170, upper bound: 0.0295170
time: 1.98 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0062520, 0.0230567, -0.0069977, 0.0207294, -0.0269814, 0.0300543
1: -0.0099608, 0.0181296, -0.0108559, 0.0159125, -0.0258733, 0.0289855
2: -0.0497237, 0.0201525, -0.0515014, 0.0146040, -0.0643277, 0.0716539
3: -0.0318634, 0.0287729, -0.0270485, 0.0303156, -0.0621790, 0.0558215
4: 0.0064942, 0.0183609, 0.0079452, 0.0179515, -0.0114573, 0.0104157
5: -0.0461198, 0.0398880, -0.0393394, 0.0427339, -0.0888536, 0.0792274
6: 0.9663780, 1.0242987, 0.9709465, 1.0261838, -0.0598058, 0.0533522
7: -0.0016273, 0.0283800, 0.0009993, 0.0298873, -0.0308473, 0.0273807
8: -0.0070478, 0.0188386, -0.0077063, 0.0167831, -0.0238308, 0.0265449
9: -0.0477295, 0.0109717, -0.0492229, 0.0063105, -0.0540400, 0.0601945

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0289664, upper bound: 0.0288972
time: 2.66 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0288566, upper bound: 0.0288566
time: 2.04 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 6.52 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 6.52
Output dim: 6, lower bound: -0.0295170, upper bound: 0.0295170
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 6.52
Output dim: 6, lower bound: -0.0295170, upper bound: 0.0295170
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 6.52
Output dim: 6, lower bound: -0.0289664, upper bound: 0.0288972
NS_A2_B2, status: Status.VERIFIED, split count: 2, time: 6.52
Output dim: 6, lower bound: -0.0288566, upper bound: 0.0288566

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0068348, 0.0207356, -0.0068348, 0.0207356, -0.0275704, 0.0275704
1: -0.0106604, 0.0159185, -0.0106604, 0.0159185, -0.0265789, 0.0265789
2: -0.0511131, 0.0146188, -0.0511131, 0.0146188, -0.0657319, 0.0657319
3: -0.0270614, 0.0299787, -0.0270614, 0.0299787, -0.0570401, 0.0570401
4: 0.0079413, 0.0179526, 0.0079413, 0.0179526, -0.0100113, 0.0100113
5: -0.0393576, 0.0421123, -0.0393576, 0.0421123, -0.0814699, 0.0814699
6: 0.9709343, 1.0257722, 0.9709343, 1.0257722, -0.0548379, 0.0548379
7: 0.0009923, 0.0295581, 0.0009923, 0.0295581, -0.0282105, 0.0282105
8: -0.0075626, 0.0167886, -0.0075626, 0.0167886, -0.0243511, 0.0243511
9: -0.0488967, 0.0063230, -0.0488967, 0.0063230, -0.0552197, 0.0552197

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0290316, upper bound: 0.0289232
time: 2.57 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0290153, upper bound: 0.0288735
time: 2.26 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0068348, 0.0207356, -0.0062520, 0.0230567, -0.0298915, 0.0269877
1: -0.0106604, 0.0159185, -0.0099608, 0.0181296, -0.0287900, 0.0258792
2: -0.0511131, 0.0146188, -0.0497237, 0.0201525, -0.0712656, 0.0643425
3: -0.0270614, 0.0299787, -0.0318634, 0.0287729, -0.0558343, 0.0618421
4: 0.0079413, 0.0179526, 0.0064942, 0.0183609, -0.0104196, 0.0114584
5: -0.0393576, 0.0421123, -0.0461198, 0.0398880, -0.0792456, 0.0882321
6: 0.9709343, 1.0257722, 0.9663780, 1.0242987, -0.0533643, 0.0593942
7: 0.0009923, 0.0295581, -0.0016273, 0.0283800, -0.0269162, 0.0306967
8: -0.0075626, 0.0167886, -0.0070478, 0.0188386, -0.0264012, 0.0238364
9: -0.0488967, 0.0063230, -0.0477295, 0.0109717, -0.0598684, 0.0540524

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0290316, upper bound: 0.0289232
time: 2.43 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0290153, upper bound: 0.0288735
time: 2.85 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 7.13 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 7.13
Output dim: 6, lower bound: -0.0290316, upper bound: 0.0289232
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 7.13
Output dim: 6, lower bound: -0.0290153, upper bound: 0.0288735
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 7.13
Output dim: 6, lower bound: -0.0290316, upper bound: 0.0289232
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 7.13
Output dim: 6, lower bound: -0.0290153, upper bound: 0.0288735

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 4.84 + 31.85 = 36.69 seconds
