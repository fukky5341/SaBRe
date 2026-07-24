## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 71.4340763181


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693)
1: (-36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435)
2: (-47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022)
3: (-53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733)
4: (-47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030)
5: (-42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557)
6: (-39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417)
7: (-45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556)
8: (-52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380)
9: (-39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.34 + 12.97 = 15.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -71.5055819, upper bound: 71.5055819

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5054969, upper bound: 71.5054971
time: 11.31 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5054904, upper bound: 71.5054904
time: 10.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 22.13 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 22.13
Output dim: 7, lower bound: -71.5054969, upper bound: 71.5054971
NS_A2, status: Status.UNKNOWN, split count: 1, time: 22.13
Output dim: 7, lower bound: -71.5054904, upper bound: 71.5054904

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -41.5313492, 33.1884651, -43.7893066, 34.9732208, -76.5045700, 76.9777603
1: -34.0791397, 29.1117401, -35.9469833, 30.6634674, -64.7426071, 65.0587234
2: -44.1147537, 27.4237118, -46.5304489, 28.8868141, -73.0015564, 73.9541626
3: -49.7571449, 24.8019276, -52.4900703, 26.1007214, -75.8578644, 77.2919998
4: -44.7353287, 34.3291016, -47.2022018, 36.1816597, -80.9169846, 81.5313034
5: -39.3197823, 30.3634129, -41.4488640, 32.0015411, -71.3213196, 71.8122787
6: -37.1415215, 37.8708420, -39.1625671, 39.9156265, -77.0571442, 77.0334091
7: -42.5678482, 31.3579025, -44.8997040, 33.0475006, -75.6153336, 76.2575989
8: -48.5502205, 33.3288956, -51.1992378, 35.1116600, -83.6618729, 84.5281219
9: -37.1215363, 37.0421677, -39.1360321, 39.0496826, -76.1712189, 76.1781998

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5030642, upper bound: 71.5029532
time: 8.33 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5031879, upper bound: 71.5031694
time: 9.95 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -43.5628090, 34.7973251, -44.3480110, 35.4147224, -78.9775314, 79.1453400
1: -35.7582169, 30.5018768, -36.4096146, 31.0495110, -66.8077240, 66.9114914
2: -46.2722740, 28.7213078, -47.1325226, 29.2586479, -75.5309219, 75.8538284
3: -52.2267685, 25.9606152, -53.1612778, 26.4302692, -78.6570358, 79.1218948
4: -46.9522934, 35.9932365, -47.8119125, 36.6408539, -83.5931320, 83.8051453
5: -41.2380905, 31.8282204, -41.9772530, 32.4084663, -73.6465454, 73.8054657
6: -38.9493599, 39.7099533, -39.6671371, 40.4197464, -79.3691025, 79.3770752
7: -44.6623039, 32.8453522, -45.4745865, 33.4798698, -78.1421738, 78.3199387
8: -50.9202843, 34.9218445, -51.8568306, 35.5580711, -86.4783554, 86.7786560
9: -38.9263306, 38.8399391, -39.6372337, 39.5488586, -78.4751740, 78.4771729

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5054904, upper bound: 71.5054904
time: 9.34 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5054904, upper bound: 71.5054904
time: 10.20 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 22.01 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.01
Output dim: 7, lower bound: -71.5030642, upper bound: 71.5029532
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.01
Output dim: 7, lower bound: -71.5031879, upper bound: 71.5031694
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.01
Output dim: 7, lower bound: -71.5054904, upper bound: 71.5054904
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.01
Output dim: 7, lower bound: -71.5054904, upper bound: 71.5054904

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -39.8919907, 31.8870411, -37.5093384, 30.0000515, -69.8920364, 69.3963699
1: -32.7115593, 27.9487247, -30.7177391, 26.2172852, -58.9288368, 58.6664619
2: -42.3106003, 26.2647362, -39.6261787, 24.4388390, -66.7494354, 65.8909149
3: -47.7924118, 23.7736301, -44.9620590, 22.1409168, -69.9333267, 68.7356873
4: -42.9599380, 32.9788895, -40.4047050, 31.0021305, -73.9620590, 73.3835907
5: -37.7640953, 29.1325397, -35.4831085, 27.2740631, -65.0381622, 64.6156464
6: -35.6427689, 36.3990326, -33.4021873, 34.2843819, -69.9271545, 69.8012161
7: -40.9134521, 29.9822235, -38.5700645, 27.7497997, -68.6632538, 68.5522690
8: -46.5824661, 31.9784927, -43.6309700, 29.9576035, -76.5400696, 75.6094666
9: -35.6163216, 35.5542564, -33.3635826, 33.3515091, -68.9678192, 68.9178238

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5025792, upper bound: 71.5023711
time: 12.64 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5030070, upper bound: 71.5028909
time: 11.44 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -39.6272354, 31.6777630, -39.3031578, 31.4253178, -71.0525513, 70.9809036
1: -32.4936829, 27.7613678, -32.2120171, 27.4565392, -59.9502220, 59.9733810
2: -42.0229416, 26.0814590, -41.5785370, 25.6539078, -67.6768494, 67.6599884
3: -47.4744911, 23.6086121, -47.1063271, 23.1988354, -70.6733170, 70.7149353
4: -42.6780930, 32.7601089, -42.3772163, 32.4698181, -75.1479034, 75.1373291
5: -37.5113258, 28.9346733, -37.1760559, 28.5977631, -66.1090851, 66.1107254
6: -35.4033585, 36.1619453, -35.0350571, 35.9041138, -71.3074722, 71.1969910
7: -40.6468506, 29.7670803, -40.3943520, 29.1929035, -69.8397369, 70.1614304
8: -46.2654724, 31.7640839, -45.7795715, 31.4035015, -77.6689758, 77.5436554
9: -35.3768120, 35.3190613, -34.9934692, 34.9634628, -70.3402710, 70.3125305

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5027165, upper bound: 71.5026591
time: 10.27 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5031297, upper bound: 71.5031073
time: 8.33 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -43.5628090, 34.7973251, -41.5313492, 33.1884651, -76.7512665, 76.3286743
1: -35.7582169, 30.5018768, -34.0791397, 29.1117401, -64.8699417, 64.5810089
2: -46.2722740, 28.7213078, -44.1147537, 27.4237118, -73.6959839, 72.8360596
3: -52.2267685, 25.9606152, -49.7571449, 24.8019276, -77.0286942, 75.7177582
4: -46.9522934, 35.9932365, -44.7353287, 34.3291016, -81.2813950, 80.7285614
5: -41.2380905, 31.8282204, -39.3197823, 30.3634129, -71.6015015, 71.1480026
6: -38.9493599, 39.7099533, -37.1415215, 37.8708420, -76.8202057, 76.8514709
7: -44.6623039, 32.8453522, -42.5678482, 31.3579025, -76.0201950, 75.4131775
8: -50.9202843, 34.9218445, -48.5502205, 33.3288956, -84.2491760, 83.4720535
9: -38.9263306, 38.8399391, -37.1215363, 37.0421677, -75.9684906, 75.9614716

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 52

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5029532, upper bound: 71.5030642
time: 8.25 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5031694, upper bound: 71.5031879
time: 10.87 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -43.5628090, 34.7973251, -43.5628090, 34.7973251, -78.3601379, 78.3601379
1: -35.7582169, 30.5018768, -35.7582169, 30.5018768, -66.2600784, 66.2600784
2: -46.2722740, 28.7213078, -46.2722740, 28.7213078, -74.9935837, 74.9935760
3: -52.2267685, 25.9606152, -52.2267685, 25.9606152, -78.1873856, 78.1873856
4: -46.9522934, 35.9932365, -46.9522934, 35.9932365, -82.9455261, 82.9455261
5: -41.2380905, 31.8282204, -41.2380905, 31.8282204, -73.0662994, 73.0662994
6: -38.9493599, 39.7099533, -38.9493599, 39.7099533, -78.6593170, 78.6593170
7: -44.6623039, 32.8453522, -44.6623039, 32.8453522, -77.5076523, 77.5076523
8: -50.9202843, 34.9218445, -50.9202843, 34.9218445, -85.8421326, 85.8421326
9: -38.9263306, 38.8399391, -38.9263306, 38.8399391, -77.7662582, 77.7662582

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 52

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5029533, upper bound: 71.5030670
time: 9.38 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5029451, upper bound: 71.5031980
time: 9.14 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 20.78 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.78
Output dim: 7, lower bound: -71.5025792, upper bound: 71.5023711
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.78
Output dim: 7, lower bound: -71.5030070, upper bound: 71.5028909
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.78
Output dim: 7, lower bound: -71.5027165, upper bound: 71.5026591
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.78
Output dim: 7, lower bound: -71.5031297, upper bound: 71.5031073
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.78
Output dim: 7, lower bound: -71.5029532, upper bound: 71.5030642
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.78
Output dim: 7, lower bound: -71.5031694, upper bound: 71.5031879
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.78
Output dim: 7, lower bound: -71.5029533, upper bound: 71.5030670
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.78
Output dim: 7, lower bound: -71.5029451, upper bound: 71.5031980

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -36.6166611, 29.3289165, -36.3839073, 29.1164856, -65.7331467, 65.7128220
1: -29.9868374, 25.7085629, -29.7824135, 25.4438686, -55.4307060, 55.4909744
2: -38.8098526, 24.0709724, -38.4178543, 23.6792507, -62.4891052, 62.4888268
3: -43.8862152, 21.8316612, -43.6223221, 21.4694862, -65.3556900, 65.4539795
4: -39.3833923, 30.3064842, -39.1821213, 30.0810261, -69.4644089, 69.4885788
5: -34.6973228, 26.7464027, -34.4239082, 26.4515190, -61.1488342, 61.1703072
6: -32.6683540, 33.4615326, -32.3780441, 33.2780609, -65.9464111, 65.8395691
7: -37.5697365, 27.4226303, -37.4257507, 26.8608398, -64.4305725, 64.8483810
8: -42.7282181, 29.3985996, -42.3024063, 29.0692749, -71.7974930, 71.7010040
9: -32.6902161, 32.6493034, -32.3570328, 32.3537064, -65.0439224, 65.0063324

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5013029, upper bound: 71.5012154
time: 10.84 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5011608, upper bound: 71.5010094
time: 12.05 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -38.4902573, 30.7902336, -37.2812119, 29.8208504, -68.3111115, 68.0714417
1: -31.5528126, 26.9952087, -30.5284958, 26.0616970, -57.6145020, 57.5237045
2: -40.8029022, 25.3103790, -39.3809471, 24.2835541, -65.0864563, 64.6913300
3: -46.1329422, 22.9346619, -44.6910706, 22.0044136, -68.1373520, 67.6257248
4: -41.4348602, 31.8377819, -40.1565857, 30.8154602, -72.2503204, 71.9943542
5: -36.4485626, 28.1094093, -35.2687073, 27.1074467, -63.5559998, 63.3781166
6: -34.3596725, 35.1527061, -33.1929321, 34.0810890, -68.4407654, 68.3456421
7: -39.4932327, 28.8602180, -38.3382225, 27.5678902, -67.0611267, 67.1984329
8: -44.9122086, 30.8694344, -43.3592186, 29.7777767, -74.6899643, 74.2286530
9: -34.3649330, 34.3062592, -33.1598701, 33.1482620, -67.5131989, 67.4661255

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 15

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5013029, upper bound: 71.5019062
time: 11.33 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5017673, upper bound: 71.5016671
time: 10.12 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -36.3620949, 29.1286583, -38.1542168, 30.5242348, -66.8863297, 67.2828674
1: -29.7761536, 25.5278568, -31.2570877, 26.6647167, -56.4408684, 56.7849426
2: -38.5333405, 23.8950615, -40.3422585, 24.8780479, -63.4113846, 64.2373199
3: -43.5798225, 21.6721973, -45.7363129, 22.5134716, -66.0932922, 67.4085007
4: -39.1124573, 30.0954418, -41.1275978, 31.5289783, -70.6414337, 71.2230377
5: -34.4537163, 26.5567265, -36.0940819, 27.7560387, -62.2097549, 62.6508026
6: -32.4375610, 33.2339287, -33.9865074, 34.8764000, -67.3139648, 67.2204285
7: -37.3128319, 27.2164497, -39.2251472, 28.2830544, -65.5958862, 66.4415970
8: -42.4242439, 29.1938705, -44.4189301, 30.4962254, -72.9204712, 73.6128006
9: -32.4602242, 32.4237747, -33.9638824, 33.9429512, -66.4031601, 66.3876572

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5027048, upper bound: 71.5026578
time: 11.46 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5027048, upper bound: 71.5026591
time: 10.12 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -38.2280197, 30.5829468, -39.0720100, 31.2439175, -69.4719391, 69.6549530
1: -31.3368073, 26.8095913, -32.0201797, 27.2984295, -58.6352386, 58.8297729
2: -40.5180931, 25.1286907, -41.3295174, 25.4965782, -66.0146713, 66.4582062
3: -45.8179245, 22.7711124, -46.8313332, 23.0605869, -68.8785095, 69.6024399
4: -41.1556702, 31.6210213, -42.1255264, 32.2807121, -73.4363861, 73.7465210
5: -36.1984024, 27.9134502, -36.9587021, 28.4286461, -64.6270447, 64.8721466
6: -34.1227646, 34.9177551, -34.8225784, 35.6980209, -69.8207855, 69.7403336
7: -39.2290421, 28.6472607, -40.1592369, 29.0081959, -68.2372360, 68.8064957
8: -44.5981216, 30.6569443, -45.5035172, 31.2211800, -75.8192978, 76.1604614
9: -34.1274147, 34.0734711, -34.7867088, 34.7572746, -68.8846893, 68.8601837

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5020701, upper bound: 71.5021130
time: 11.70 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5019187, upper bound: 71.5019256
time: 11.74 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -37.2833023, 29.8234921, -39.8919907, 31.8870411, -69.1703262, 69.7154770
1: -30.5295372, 26.0562668, -32.7115593, 27.9487247, -58.4782600, 58.7678261
2: -39.3682632, 24.2723541, -42.3106003, 26.2647362, -65.6329956, 66.5829544
3: -44.6983490, 22.0011654, -47.7924118, 23.7736301, -68.4719772, 69.7935791
4: -40.1551437, 30.8126030, -42.9599380, 32.9788895, -73.1340332, 73.7725372
5: -35.2735596, 27.0988941, -37.7640953, 29.1325397, -64.4060974, 64.8629913
6: -33.1885643, 34.0789032, -35.6427689, 36.3990326, -69.5876007, 69.7216644
7: -38.3323021, 27.5465565, -40.9134521, 29.9822235, -68.3145142, 68.4600067
8: -43.3506355, 29.7709503, -46.5824661, 31.9784927, -75.3291245, 76.3534164
9: -33.1536064, 33.1420403, -35.6163216, 35.5542564, -68.7078629, 68.7583618

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5023711, upper bound: 71.5025792
time: 10.21 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5028909, upper bound: 71.5030070
time: 10.94 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -39.1304054, 31.2911301, -39.6272354, 31.6777630, -70.8081665, 70.9183655
1: -32.0676727, 27.3330803, -32.4936829, 27.7613678, -59.8290329, 59.8267593
2: -41.3805656, 25.5260773, -42.0229416, 26.0814590, -67.4620132, 67.5490189
3: -46.9044724, 23.0917473, -47.4744911, 23.6086121, -70.5130844, 70.5662308
4: -42.1866531, 32.3242149, -42.6780930, 32.7601089, -74.9467545, 75.0023041
5: -37.0160599, 28.4633408, -37.5113258, 28.9346733, -65.9507217, 65.9746552
6: -34.8709679, 35.7472229, -35.4033585, 36.1619453, -71.0329132, 71.1505814
7: -40.2123642, 29.0351772, -40.6468506, 29.7670803, -69.9794312, 69.6820221
8: -45.5640869, 31.2603073, -46.2654724, 31.7640839, -77.3281631, 77.5257797
9: -34.8320007, 34.8024597, -35.3768120, 35.3190613, -70.1510620, 70.1792679

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5026591, upper bound: 71.5027165
time: 11.45 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5031073, upper bound: 71.5031297
time: 14.50 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -37.2833023, 29.8234921, -41.8955040, 33.4720268, -70.7553177, 71.7189941
1: -30.5295372, 26.0562668, -34.3674088, 29.3180466, -59.8475838, 60.4236717
2: -39.3682632, 24.2723541, -44.4354095, 27.5403137, -66.9085770, 68.7077637
3: -44.6983490, 22.0011654, -50.2259521, 24.9139385, -69.6122894, 72.2271042
4: -40.1551437, 30.8126030, -45.1441460, 34.6189384, -74.7740784, 75.9567490
5: -35.2735596, 27.0988941, -39.6534500, 30.5775414, -65.8511047, 66.7523422
6: -33.1885643, 34.0789032, -37.4178734, 38.2119789, -71.4005432, 71.4967651
7: -38.3323021, 27.5465565, -42.9759636, 31.4446030, -69.7769012, 70.5225067
8: -43.3506355, 29.7709503, -48.9138985, 33.5453339, -76.8959503, 78.6848450
9: -33.1536064, 33.1420403, -37.3929176, 37.3229904, -70.4765930, 70.5349579

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5023786, upper bound: 71.5026032
time: 8.95 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5028918, upper bound: 71.5030095
time: 9.44 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -39.1304054, 31.2911301, -41.5986862, 33.2368813, -72.3672867, 72.8898087
1: -32.0676727, 27.3330803, -34.1228409, 29.1100979, -61.1777725, 61.4559212
2: -41.3805656, 25.5260773, -44.1140709, 27.3366547, -68.7172165, 69.6401520
3: -46.9044724, 23.0917473, -49.8678360, 24.7295914, -71.6340637, 72.9595795
4: -42.1866531, 32.3242149, -44.8287086, 34.3746223, -76.5612793, 77.1529160
5: -37.0160599, 28.4633408, -39.3692055, 30.3586178, -67.3746643, 67.8325500
6: -34.8709679, 35.7472229, -37.1498642, 37.9467392, -72.8176956, 72.8970871
7: -40.2123642, 29.0351772, -42.6785431, 31.2074699, -71.4198303, 71.7137146
8: -45.5640869, 31.2603073, -48.5607605, 33.3057785, -78.8698654, 79.8210678
9: -34.8320007, 34.8024597, -37.1260834, 37.0602112, -71.8922119, 71.9285431

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5023786, upper bound: 71.5027582
time: 11.01 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5031334, upper bound: 71.5031358
time: 8.61 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 21.86 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 7, lower bound: -71.5013029, upper bound: 71.5012154
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 7, lower bound: -71.5011608, upper bound: 71.5010094
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 7, lower bound: -71.5013029, upper bound: 71.5019062
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 7, lower bound: -71.5017673, upper bound: 71.5016671
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 7, lower bound: -71.5027048, upper bound: 71.5026578
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 7, lower bound: -71.5027048, upper bound: 71.5026591
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 7, lower bound: -71.5020701, upper bound: 71.5021130
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 7, lower bound: -71.5019187, upper bound: 71.5019256
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 7, lower bound: -71.5023711, upper bound: 71.5025792
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 7, lower bound: -71.5028909, upper bound: 71.5030070
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 7, lower bound: -71.5026591, upper bound: 71.5027165
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 7, lower bound: -71.5031073, upper bound: 71.5031297
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 7, lower bound: -71.5023786, upper bound: 71.5026032
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 7, lower bound: -71.5028918, upper bound: 71.5030095
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 7, lower bound: -71.5023786, upper bound: 71.5027582
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 7, lower bound: -71.5031334, upper bound: 71.5031358

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -36.1495857, 28.9554195, -33.3586388, 26.7015381, -62.8511200, 62.3140564
1: -29.5970898, 25.3801079, -27.2656116, 23.3279953, -52.9250870, 52.6457214
2: -38.2939453, 23.7361336, -35.0846939, 21.5307293, -59.8246765, 58.8208275
3: -43.3324356, 21.5334721, -40.0344810, 19.5604706, -62.8929062, 61.5679550
4: -38.8818779, 29.9218140, -35.9444580, 27.5956173, -66.4774933, 65.8662720
5: -34.2511902, 26.3940430, -31.5467758, 24.1797066, -58.4308929, 57.9408150
6: -32.2421341, 33.0402985, -29.6287632, 30.5562649, -62.7984009, 62.6690598
7: -37.1033096, 27.0243874, -34.4043961, 24.3001842, -61.4034958, 61.4287834
8: -42.1569443, 29.0118809, -38.6190529, 26.5692616, -68.7262039, 67.6309280
9: -32.2633591, 32.2257996, -29.5970402, 29.6180058, -61.8813591, 61.8228340

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5011607, upper bound: 71.5010094
time: 10.15 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5011607, upper bound: 71.5010094
time: 9.13 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -35.1857338, 28.1859112, -34.7947121, 27.8233776, -63.0091095, 62.9806213
1: -28.7932415, 24.7094097, -28.4764462, 24.3344936, -53.1277351, 53.1858559
2: -37.2373962, 23.0529556, -36.6173325, 22.4178467, -59.6552429, 59.6702843
3: -42.1930428, 20.9274807, -41.8022614, 20.3609486, -62.5539932, 62.7297401
4: -37.8489914, 29.1314774, -37.5517960, 28.7808552, -66.6298447, 66.6832657
5: -33.3345413, 25.6715755, -32.9121132, 25.2103920, -58.5449219, 58.5836868
6: -31.3673592, 32.1725960, -30.9260540, 31.8544579, -63.2217979, 63.0986481
7: -36.1400642, 26.2162075, -35.9195404, 25.3308945, -61.4709587, 62.1357498
8: -40.9854279, 28.2183590, -40.2929802, 27.6818466, -68.6672745, 68.5113373
9: -31.3870144, 31.3525295, -30.8742828, 30.8772812, -62.2642860, 62.2268105

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4975002, upper bound: 71.4969975
time: 9.93 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4975002, upper bound: 71.4995938
time: 10.82 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -38.0103683, 30.4065742, -34.2080688, 27.3664989, -65.3768539, 64.6146393
1: -31.1535759, 26.6592045, -27.9722843, 23.9131908, -55.0667648, 54.6314888
2: -40.2737122, 24.9666328, -35.9956894, 22.1004944, -62.3742065, 60.9623184
3: -45.5655441, 22.6296806, -41.0451088, 20.0668182, -65.6323547, 63.6747894
4: -40.9203224, 31.4429111, -36.8672943, 28.2903004, -69.2106094, 68.3102036
5: -35.9917679, 27.7474537, -32.3447151, 24.7990799, -60.7908478, 60.0921707
6: -33.9228287, 34.7205963, -30.4003887, 31.3166485, -65.2394791, 65.1209793
7: -39.0145836, 28.4511814, -35.2672157, 24.9685059, -63.9830818, 63.7183952
8: -44.3256607, 30.4706802, -39.6181488, 27.2401009, -71.5657501, 70.0888214
9: -33.9259415, 33.8718338, -30.3567162, 30.3683052, -64.2942505, 64.2285385

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 94

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5017674, upper bound: 71.5016671
time: 9.25 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5017674, upper bound: 71.5016671
time: 9.98 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -37.0128059, 29.6119423, -35.6555862, 28.4943237, -65.5071259, 65.2675323
1: -30.3239307, 25.9662189, -29.1916733, 24.9244251, -55.2483521, 55.1578903
2: -39.1814995, 24.2598419, -37.5381165, 22.9929810, -62.1744728, 61.7979584
3: -44.3855820, 22.0045605, -42.8281822, 20.8741913, -65.2597733, 64.8327408
4: -39.8523865, 30.6268692, -38.4815979, 29.4862518, -69.3386383, 69.1084671
5: -35.0445290, 27.0011902, -33.7217293, 25.8382988, -60.8828239, 60.7229195
6: -33.0194321, 33.8243484, -31.7122269, 32.6227951, -65.6422195, 65.5365753
7: -38.0201263, 27.6141701, -36.7942772, 26.0086594, -64.0287857, 64.4084473
8: -43.1157608, 29.6516609, -41.3042831, 28.3589401, -71.4747009, 70.9559479
9: -33.0190125, 32.9696579, -31.6430321, 31.6375694, -64.6565857, 64.6126862

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4980511, upper bound: 71.4977727
time: 10.95 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5004617, upper bound: 71.5003801
time: 11.91 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -36.3620949, 29.1286583, -35.9377937, 28.7750282, -65.1371231, 65.0664520
1: -29.7761536, 25.5278568, -29.4249897, 25.1437912, -54.9199371, 54.9528465
2: -38.5333405, 23.8950615, -37.9770050, 23.4499245, -61.9832649, 61.8720665
3: -43.5798225, 21.6721973, -43.0581169, 21.2474689, -64.8272934, 64.7303009
4: -39.1124573, 30.0954418, -38.7089157, 29.7082596, -68.8207169, 68.8043594
5: -34.4537163, 26.5567265, -34.0088539, 26.1468925, -60.6006088, 60.5655823
6: -32.4375610, 33.2339287, -32.0121689, 32.8703537, -65.3079147, 65.2460938
7: -37.3128319, 27.2164497, -36.9364166, 26.6332054, -63.9460373, 64.1528625
8: -42.4242439, 29.1938705, -41.8283348, 28.7577972, -71.1820374, 71.0222015
9: -32.4602242, 32.4237747, -31.9901371, 31.9801216, -64.4403305, 64.4139099

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5015611, upper bound: 71.5014123
time: 12.51 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5012374, upper bound: 71.5011599
time: 11.19 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -36.3620949, 29.1286583, -37.9843216, 30.3919678, -66.7540588, 67.1129761
1: -29.7761536, 25.5278568, -31.1150589, 26.5430222, -56.3191719, 56.6429138
2: -38.5333405, 23.8950615, -40.1473618, 24.7519054, -63.2852402, 64.0424194
3: -43.5798225, 21.6721973, -45.5378571, 22.4079590, -65.9877777, 67.2100372
4: -39.1124573, 30.0954418, -40.9400978, 31.3855000, -70.4979553, 71.0355377
5: -34.4537163, 26.5567265, -35.9366455, 27.6234760, -62.0771828, 62.4933701
6: -32.4375610, 33.2339287, -33.8250389, 34.7217712, -67.1593323, 67.0589600
7: -37.3128319, 27.2164497, -39.0459862, 28.1274128, -65.4402390, 66.2624207
8: -42.4242439, 29.1938705, -44.2067566, 30.3553772, -72.7796173, 73.4006271
9: -32.4602242, 32.4237747, -33.8049469, 33.7844505, -66.2446671, 66.2287216

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5015611, upper bound: 71.5015180
time: 11.19 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5012374, upper bound: 71.5012212
time: 9.82 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -37.7498169, 30.2011528, -35.9202156, 28.7262230, -66.4760361, 66.1213684
1: -30.9390774, 26.4746113, -29.3963928, 25.0910625, -56.0301361, 55.8710022
2: -39.9906731, 24.7866287, -37.8536263, 23.2472954, -63.2379684, 62.6402550
3: -45.2520561, 22.4670467, -43.0889244, 21.0649128, -66.3169708, 65.5559692
4: -40.6430702, 31.2281075, -38.7478142, 29.6888180, -70.3318863, 69.9759140
5: -35.7432556, 27.5529594, -33.9569473, 26.0514545, -61.7947083, 61.5099030
6: -33.6874962, 34.4872971, -31.9493332, 32.8640594, -66.5515518, 66.4366302
7: -38.7523232, 28.2400551, -37.0084877, 26.3271408, -65.0794678, 65.2485428
8: -44.0141068, 30.2604408, -41.6590576, 28.6171322, -72.6312408, 71.9194946
9: -33.6905327, 33.6405220, -31.9050026, 31.9012604, -65.5917969, 65.5455170

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 94

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5019187, upper bound: 71.5019256
time: 10.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5019187, upper bound: 71.5019256
time: 8.75 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -36.7598495, 29.4128132, -37.3842773, 29.8676605, -66.6275024, 66.7970886
1: -30.1159267, 25.7870331, -30.6305275, 26.1169262, -56.2328529, 56.4175606
2: -38.9071503, 24.0856571, -39.4226646, 24.1629143, -63.0700645, 63.5083122
3: -44.0811157, 21.8470592, -44.8934326, 21.8951797, -65.9762878, 66.7404938
4: -39.5829620, 30.4182167, -40.3762665, 30.9011879, -70.4841461, 70.7944717
5: -34.8030052, 26.8126431, -35.3527260, 27.1111889, -61.9141922, 62.1653671
6: -32.7910233, 33.5983620, -33.2836952, 34.1836967, -66.9747086, 66.8820572
7: -37.7652626, 27.4098930, -38.5493011, 27.4008141, -65.1660767, 65.9591980
8: -42.8137894, 29.4479942, -43.3724251, 29.7538395, -72.5676117, 72.8204041
9: -32.7909622, 32.7453384, -33.2160759, 33.1888885, -65.9798508, 65.9614029

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4980875, upper bound: 71.4978840
time: 15.49 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5005726, upper bound: 71.5005747
time: 11.56 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -36.1619225, 28.9432755, -36.6166611, 29.3289165, -65.4908371, 65.5599365
1: -29.5975132, 25.2853031, -29.9868374, 25.7085629, -55.3060684, 55.2721367
2: -38.1639328, 23.5167027, -38.8098526, 24.0709724, -62.2348976, 62.3265533
3: -43.3639565, 21.3320808, -43.8862152, 21.8316612, -65.1956177, 65.2182922
4: -38.9365158, 29.8949642, -39.3833923, 30.3064842, -69.2429886, 69.2783585
5: -34.2183800, 26.2795258, -34.6973228, 26.7464027, -60.9647751, 60.9768372
6: -32.1684074, 33.0758209, -32.6683540, 33.4615326, -65.6299362, 65.7441711
7: -37.1917419, 26.6613274, -37.5697365, 27.4226303, -64.6143723, 64.2310638
8: -42.0271416, 28.8857918, -42.7282181, 29.3985996, -71.4257355, 71.6140060
9: -32.1511116, 32.1480179, -32.6902161, 32.6493034, -64.8004150, 64.8382187

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5012154, upper bound: 71.5013029
time: 10.49 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5010094, upper bound: 71.5011608
time: 11.05 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -37.0555267, 29.6445656, -38.4902573, 30.7902336, -67.8457489, 68.1348267
1: -30.3405914, 25.9008675, -31.5528126, 26.9952087, -57.3357925, 57.4536781
2: -39.1233330, 24.1175308, -40.8029022, 25.3103790, -64.4337158, 64.9204330
3: -44.4278183, 21.8648491, -46.1329422, 22.9346619, -67.3624802, 67.9977875
4: -39.9073868, 30.6262779, -41.4348602, 31.8377819, -71.7451706, 72.0611420
5: -35.0595284, 26.9325600, -36.4485626, 28.1094093, -63.1689301, 63.3811188
6: -32.9797096, 33.8758698, -34.3596725, 35.1527061, -68.1324158, 68.2355347
7: -38.1007729, 27.3650742, -39.4932327, 28.8602180, -66.9609833, 66.8582993
8: -43.0793152, 29.5914249, -44.9122086, 30.8694344, -73.9487457, 74.5036316
9: -32.9503326, 32.9390984, -34.3649330, 34.3062592, -67.2565918, 67.3040314

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5019062, upper bound: 71.5019361
time: 10.67 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5016671, upper bound: 71.5017674
time: 10.27 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 15.30 + 586.90 = 602.21 seconds
