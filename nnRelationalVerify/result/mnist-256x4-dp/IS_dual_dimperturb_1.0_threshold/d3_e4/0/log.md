## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 27.1733048946


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=93, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570)
1: (-16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204)
2: (-27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886)
3: (-24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909)
4: (-24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434)
5: (-18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721)
6: (-19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750)
7: (-22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555)
8: (-25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507)
9: (-17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.45 + 6.44 = 7.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -27.2005054, upper bound: 27.2005054

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1972990, upper bound: 27.1969081
time: 5.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987716, upper bound: 27.1987716
time: 5.21 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.88 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.88
Output dim: 2, lower bound: -27.1972990, upper bound: 27.1969081
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.88
Output dim: 2, lower bound: -27.1987716, upper bound: 27.1987716

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -17.7370834, 13.4382677, -19.5840302, 14.8673496, -32.6044312, 33.0222969
1: -14.1783390, 11.9624004, -15.7014399, 13.2516689, -27.4300079, 27.6638412
2: -23.5818024, 7.6063967, -25.8618412, 8.7477245, -32.3295212, 33.4682388
3: -21.0418549, 9.4901848, -23.2685165, 10.5544968, -31.5963516, 32.7587013
4: -21.2769165, 12.8763294, -23.4271584, 14.2579031, -35.5348206, 36.3034897
5: -15.8263702, 13.6713247, -17.5454559, 15.0845242, -30.9108944, 31.2167816
6: -16.8846321, 14.4183712, -18.7129440, 15.9619741, -32.8466072, 33.1313171
7: -19.0637054, 13.8504381, -21.0339527, 15.3976727, -34.4613762, 34.8843880
8: -21.9471779, 12.5008354, -24.2660751, 13.8591270, -35.8062973, 36.7669067
9: -15.2502155, 17.9744301, -16.8939228, 19.7750587, -35.0252762, 34.8683472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=92, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=34, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=249, inp2_unstable=253, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1966152, upper bound: 27.1966152
time: 8.25 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1966152, upper bound: 27.1969081
time: 4.48 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -19.2813759, 14.6328573, -20.0912285, 15.2757206, -34.5570946, 34.7240829
1: -15.4504633, 13.0406036, -16.1221123, 13.6216984, -29.0721626, 29.1627159
2: -25.4806252, 8.5678139, -26.4521561, 9.1441746, -34.6247940, 35.0199623
3: -22.9028149, 10.3804493, -23.8861313, 10.8818111, -33.7846260, 34.2665787
4: -23.0672989, 14.0238628, -24.0047455, 14.6549339, -37.7222328, 38.0286064
5: -17.2664795, 14.8502493, -18.0368500, 15.4769497, -32.7434311, 32.8871002
6: -18.4147415, 15.7105303, -19.2292881, 16.3985214, -34.8132591, 34.9398193
7: -20.7114697, 15.1501713, -21.5708580, 15.8566313, -36.5681000, 36.7210312
8: -23.8846569, 13.6342249, -24.8998566, 14.2604790, -38.1451340, 38.5340805
9: -16.6258526, 19.4761505, -17.3630238, 20.2495880, -36.8754425, 36.8391685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=91, inp2_unstable=92, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=39, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=253, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1969081, upper bound: 27.1972990
time: 27.76 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1969081, upper bound: 27.1987716
time: 4.19 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 33.45 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 33.45
Output dim: 2, lower bound: -27.1966152, upper bound: 27.1966152
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 33.45
Output dim: 2, lower bound: -27.1966152, upper bound: 27.1969081
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 33.45
Output dim: 2, lower bound: -27.1969081, upper bound: 27.1972990
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 33.45
Output dim: 2, lower bound: -27.1969081, upper bound: 27.1987716

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -17.7370834, 13.4382677, -17.7370834, 13.4382677, -31.1753502, 31.1753502
1: -14.1783390, 11.9624004, -14.1783390, 11.9624004, -26.1407394, 26.1407394
2: -23.5818024, 7.6063967, -23.5818024, 7.6063967, -31.1881962, 31.1881981
3: -21.0418549, 9.4901848, -21.0418549, 9.4901848, -30.5320396, 30.5320396
4: -21.2769165, 12.8763294, -21.2769165, 12.8763294, -34.1532440, 34.1532440
5: -15.8263702, 13.6713247, -15.8263702, 13.6713247, -29.4976959, 29.4976959
6: -16.8846321, 14.4183712, -16.8846321, 14.4183712, -31.3030014, 31.3030033
7: -19.0637054, 13.8504381, -19.0637054, 13.8504381, -32.9141426, 32.9141426
8: -21.9471779, 12.5008354, -21.9471779, 12.5008354, -34.4480057, 34.4480057
9: -15.2502155, 17.9744301, -15.2502155, 17.9744301, -33.2246475, 33.2246475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=89, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=249, inp2_unstable=249, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1866518, upper bound: 27.1856288
time: 6.06 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1953966, upper bound: 27.1953966
time: 4.74 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -17.7370834, 13.4382677, -19.2813759, 14.6328573, -32.3699417, 32.7196426
1: -14.1783390, 11.9624004, -15.4504633, 13.0406036, -27.2189426, 27.4128647
2: -23.5818024, 7.6063967, -25.4806252, 8.5678139, -32.1496048, 33.0870132
3: -21.0418549, 9.4901848, -22.9028149, 10.3804493, -31.4223042, 32.3929977
4: -21.2769165, 12.8763294, -23.0672989, 14.0238628, -35.3007812, 35.9436264
5: -15.8263702, 13.6713247, -17.2664795, 14.8502493, -30.6766167, 30.9378052
6: -16.8846321, 14.4183712, -18.4147415, 15.7105303, -32.5951614, 32.8331146
7: -19.0637054, 13.8504381, -20.7114697, 15.1501713, -34.2138748, 34.5619049
8: -21.9471779, 12.5008354, -23.8846569, 13.6342249, -35.5813980, 36.3854866
9: -15.2502155, 17.9744301, -16.6258526, 19.4761505, -34.7263641, 34.6002808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=31, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=249, inp2_unstable=253, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1856288, upper bound: 27.1866688
time: 3.84 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1953966, upper bound: 27.1956661
time: 4.13 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -19.2813759, 14.6328573, -17.7370834, 13.4382677, -32.7196388, 32.3699417
1: -15.4504633, 13.0406036, -14.1783390, 11.9624004, -27.4128647, 27.2189426
2: -25.4806252, 8.5678139, -23.5818024, 7.6063967, -33.0870171, 32.1496124
3: -22.9028149, 10.3804493, -21.0418549, 9.4901848, -32.3929977, 31.4223042
4: -23.0672989, 14.0238628, -21.2769165, 12.8763294, -35.9436264, 35.3007812
5: -17.2664795, 14.8502493, -15.8263702, 13.6713247, -30.9378052, 30.6766186
6: -18.4147415, 15.7105303, -16.8846321, 14.4183712, -32.8331146, 32.5951614
7: -20.7114697, 15.1501713, -19.0637054, 13.8504381, -34.5619087, 34.2138748
8: -23.8846569, 13.6342249, -21.9471779, 12.5008354, -36.3854866, 35.5813980
9: -16.6258526, 19.4761505, -15.2502155, 17.9744301, -34.6002808, 34.7263641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=91, inp2_unstable=89, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=253, inp2_unstable=249, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1866688, upper bound: 27.1857508
time: 5.00 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1956661, upper bound: 27.1960934
time: 4.79 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -19.2813759, 14.6328573, -19.2813759, 14.6328573, -33.9142342, 33.9142342
1: -15.4504633, 13.0406036, -15.4504633, 13.0406036, -28.4910660, 28.4910660
2: -25.4806252, 8.5678139, -25.4806252, 8.5678139, -34.0484314, 34.0484276
3: -22.9028149, 10.3804493, -22.9028149, 10.3804493, -33.2832565, 33.2832565
4: -23.0672989, 14.0238628, -23.0672989, 14.0238628, -37.0911636, 37.0911636
5: -17.2664795, 14.8502493, -17.2664795, 14.8502493, -32.1167297, 32.1167297
6: -18.4147415, 15.7105303, -18.4147415, 15.7105303, -34.1252670, 34.1252708
7: -20.7114697, 15.1501713, -20.7114697, 15.1501713, -35.8616409, 35.8616409
8: -23.8846569, 13.6342249, -23.8846569, 13.6342249, -37.5188828, 37.5188789
9: -16.6258526, 19.4761505, -16.6258526, 19.4761505, -36.1020050, 36.1020050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=91, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=253, inp2_unstable=253, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1859429, upper bound: 27.1876655
time: 5.35 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1956661, upper bound: 27.1976968
time: 4.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 11.66 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 11.66
Output dim: 2, lower bound: -27.1866518, upper bound: 27.1856288
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 11.66
Output dim: 2, lower bound: -27.1953966, upper bound: 27.1953966
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 11.66
Output dim: 2, lower bound: -27.1856288, upper bound: 27.1866688
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 11.66
Output dim: 2, lower bound: -27.1953966, upper bound: 27.1956661
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 11.66
Output dim: 2, lower bound: -27.1866688, upper bound: 27.1857508
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 11.66
Output dim: 2, lower bound: -27.1956661, upper bound: 27.1960934
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 11.66
Output dim: 2, lower bound: -27.1859429, upper bound: 27.1876655
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 11.66
Output dim: 2, lower bound: -27.1956661, upper bound: 27.1976968

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -16.8317299, 12.7335157, -16.6347942, 12.5983982, -29.4301281, 29.3683090
1: -13.4129200, 11.3248215, -13.2749472, 11.2176657, -24.6305809, 24.5997696
2: -22.4660797, 6.9903717, -22.1815758, 7.0285330, -29.4946136, 29.1719475
3: -19.9012680, 8.9243526, -19.7020493, 8.8759098, -28.7771759, 28.6263962
4: -20.2617760, 12.1930265, -19.9911423, 12.0760460, -32.3378220, 32.1841660
5: -14.9546680, 12.9803066, -14.8098221, 12.8325138, -27.7871799, 27.7901249
6: -16.0028496, 13.6762362, -15.8234844, 13.5238495, -29.5266991, 29.4997158
7: -18.0861378, 13.0791740, -17.8823700, 12.9650517, -31.0511875, 30.9615440
8: -20.8303032, 11.8382320, -20.5825214, 11.7246342, -32.5549393, 32.4207535
9: -14.4516153, 17.0915165, -14.2965860, 16.8806686, -31.3322830, 31.3881035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=89, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=12, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=246, inp2_unstable=246, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1844891, upper bound: 27.1844891
time: 13.06 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1844891, upper bound: 27.1856288
time: 6.23 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -17.3379898, 13.1264954, -17.7370834, 13.4382677, -30.7762566, 30.8635788
1: -13.8477888, 11.6870747, -14.1783390, 11.9624004, -25.8101883, 25.8654137
2: -23.0851631, 7.3696795, -23.5818024, 7.6063967, -30.6915569, 30.9514809
3: -20.5558205, 9.2606688, -21.0418549, 9.4901848, -30.0460052, 30.3025208
4: -20.8129292, 12.5806046, -21.2769165, 12.8763294, -33.6892586, 33.8575211
5: -15.4483376, 13.3657217, -15.8263702, 13.6713247, -29.1196632, 29.1920910
6: -16.4954891, 14.0895319, -16.8846321, 14.4183712, -30.9138565, 30.9741611
7: -18.6347389, 13.5205002, -19.0637054, 13.8504381, -32.4851761, 32.5842056
8: -21.4465790, 12.2142963, -21.9471779, 12.5008354, -33.9474144, 34.1614647
9: -14.9008160, 17.5803413, -15.2502155, 17.9744301, -32.8752441, 32.8305588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=89, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=249, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1856288, upper bound: 27.1866518
time: 4.87 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1856288, upper bound: 27.1953966
time: 5.64 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -16.6347942, 12.5983982, -17.8874741, 13.5268631, -30.1616573, 30.4858723
1: -13.2749472, 11.2176657, -14.2625885, 12.0275040, -25.3024521, 25.4802494
2: -22.1815758, 7.0285330, -23.8239956, 7.5214262, -29.7030029, 30.8525257
3: -19.7020493, 8.8759098, -21.1708260, 9.4986572, -29.2006989, 30.0467358
4: -19.9911423, 12.0760460, -21.4935303, 12.9550762, -32.9462166, 33.5695763
5: -14.8098221, 12.8325138, -15.9107056, 13.7824078, -28.5922279, 28.7432117
6: -15.8234844, 13.5238495, -17.0131073, 14.5267038, -30.3501854, 30.5369568
7: -17.8823700, 12.9650517, -19.2299156, 13.9158087, -31.7981796, 32.1949615
8: -20.5825214, 11.7246342, -22.1316624, 12.5717964, -33.1543159, 33.8562927
9: -14.2965860, 16.8806686, -15.3638668, 18.1342430, -32.4308243, 32.2445374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=246, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1847268, upper bound: 27.1846315
time: 5.09 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1847268, upper bound: 27.1866688
time: 4.57 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -17.7370834, 13.4382677, -18.7389374, 14.2121658, -31.9492493, 32.1772041
1: -14.1783390, 11.9624004, -14.9956312, 12.6541805, -26.8325195, 26.9580307
2: -23.5818024, 7.6063967, -24.8272076, 8.2060719, -31.7878704, 32.4336014
3: -21.0418549, 9.4901848, -22.2329102, 10.0517826, -31.0936375, 31.7230949
4: -21.2769165, 12.8763294, -22.4428482, 13.6119614, -34.8888779, 35.3191757
5: -15.8263702, 13.6713247, -16.7553368, 14.4334555, -30.2598267, 30.4266624
6: -16.8846321, 14.4183712, -17.8754063, 15.2532148, -32.1378479, 32.2937775
7: -19.0637054, 13.8504381, -20.1374226, 14.6879034, -33.7516098, 33.9878616
8: -21.9471779, 12.5008354, -23.2031250, 13.2296448, -35.1768150, 35.7039566
9: -15.2502155, 17.9744301, -16.1392937, 18.9533577, -34.2035751, 34.1137199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=23, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=249, inp2_unstable=251, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1871831, upper bound: 27.1859430
time: 6.89 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1871831, upper bound: 27.1956661
time: 3.68 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -17.8874741, 13.5268631, -16.6347942, 12.5983982, -30.4858723, 30.1616573
1: -14.2625885, 12.0275040, -13.2749472, 11.2176657, -25.4802513, 25.3024521
2: -23.8239956, 7.5214262, -22.1815758, 7.0285330, -30.8525276, 29.7030010
3: -21.1708260, 9.4986572, -19.7020493, 8.8759098, -30.0467358, 29.2007027
4: -21.4935303, 12.9550762, -19.9911423, 12.0760460, -33.5695763, 32.9462204
5: -15.9107056, 13.7824078, -14.8098221, 12.8325138, -28.7432117, 28.5922260
6: -17.0131073, 14.5267038, -15.8234844, 13.5238495, -30.5369568, 30.3501835
7: -19.2299156, 13.9158087, -17.8823700, 12.9650517, -32.1949615, 31.7981796
8: -22.1316624, 12.5717964, -20.5825214, 11.7246342, -33.8562965, 33.1543159
9: -15.3638668, 18.1342430, -14.2965860, 16.8806686, -32.2445374, 32.4308281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=91, inp2_unstable=89, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=246, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1846315, upper bound: 27.1847268
time: 3.24 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1846315, upper bound: 27.1857508
time: 3.83 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -18.7389374, 14.2121658, -17.7370834, 13.4382677, -32.1772041, 31.9492493
1: -14.9956312, 12.6541805, -14.1783390, 11.9624004, -26.9580307, 26.8325195
2: -24.8272076, 8.2060719, -23.5818024, 7.6063967, -32.4336014, 31.7878742
3: -22.2329102, 10.0517826, -21.0418549, 9.4901848, -31.7230949, 31.0936375
4: -22.4428482, 13.6119614, -21.2769165, 12.8763294, -35.3191757, 34.8888779
5: -16.7553368, 14.4334555, -15.8263702, 13.6713247, -30.4266624, 30.2598267
6: -17.8754063, 15.2532148, -16.8846321, 14.4183712, -32.2937775, 32.1378479
7: -20.1374226, 14.6879034, -19.0637054, 13.8504381, -33.9878616, 33.7516098
8: -23.2031250, 13.2296448, -21.9471779, 12.5008354, -35.7039566, 35.1768150
9: -16.1392937, 18.9533577, -15.2502155, 17.9744301, -34.1137238, 34.2035675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=89, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=17, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=249, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1859429, upper bound: 27.1871831
time: 6.74 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1859429, upper bound: 27.1960934
time: 5.14 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -17.9179478, 13.5797911, -17.8874741, 13.5268631, -31.4447994, 31.4672661
1: -14.3123016, 12.0825882, -14.2625885, 12.0275040, -26.3398056, 26.3451710
2: -23.7918434, 7.7346840, -23.8239956, 7.5214262, -31.3132706, 31.5586796
3: -21.2268486, 9.5862627, -21.1708260, 9.4986572, -30.7254982, 30.7570877
4: -21.4794617, 13.0140543, -21.4935303, 12.9550762, -34.4345360, 34.5075836
5: -15.9881611, 13.8063536, -15.9107056, 13.7824078, -29.7705688, 29.7170582
6: -17.0717278, 14.5661659, -17.0131073, 14.5267038, -31.5984306, 31.5792732
7: -19.2609653, 14.0104275, -19.2299156, 13.9158087, -33.1767731, 33.2403336
8: -22.1756287, 12.6353064, -22.1316624, 12.5717964, -34.7474213, 34.7669678
9: -15.4163799, 18.1387329, -15.3638668, 18.1342430, -33.5506210, 33.5026016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1852035, upper bound: 27.1852029
time: 3.87 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1852035, upper bound: 27.1876654
time: 5.13 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -19.2813759, 14.6328573, -18.7389374, 14.2121658, -33.4935417, 33.3717957
1: -15.4504633, 13.0406036, -14.9956312, 12.6541805, -28.1046448, 28.0362358
2: -25.4806252, 8.5678139, -24.8272076, 8.2060719, -33.6866875, 33.3950195
3: -22.9028149, 10.3804493, -22.2329102, 10.0517826, -32.9545975, 32.6133537
4: -23.0672989, 14.0238628, -22.4428482, 13.6119614, -36.6792603, 36.4667130
5: -17.2664795, 14.8502493, -16.7553368, 14.4334555, -31.6999359, 31.6055870
6: -18.4147415, 15.7105303, -17.8754063, 15.2532148, -33.6679573, 33.5859375
7: -20.7114697, 15.1501713, -20.1374226, 14.6879034, -35.3993721, 35.2875938
8: -23.8846569, 13.6342249, -23.2031250, 13.2296448, -37.1142921, 36.8373489
9: -16.6258526, 19.4761505, -16.1392937, 18.9533577, -35.5792084, 35.6154366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=91, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=23, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=253, inp2_unstable=251, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1874860, upper bound: 27.1864186
time: 7.16 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1874860, upper bound: 27.1976968
time: 6.38 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 15.10 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 2, lower bound: -27.1844891, upper bound: 27.1844891
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 2, lower bound: -27.1844891, upper bound: 27.1856288
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 2, lower bound: -27.1856288, upper bound: 27.1866518
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 2, lower bound: -27.1856288, upper bound: 27.1953966
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 2, lower bound: -27.1847268, upper bound: 27.1846315
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 2, lower bound: -27.1847268, upper bound: 27.1866688
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 2, lower bound: -27.1871831, upper bound: 27.1859430
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 2, lower bound: -27.1871831, upper bound: 27.1956661
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 2, lower bound: -27.1846315, upper bound: 27.1847268
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 2, lower bound: -27.1846315, upper bound: 27.1857508
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 2, lower bound: -27.1859429, upper bound: 27.1871831
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 2, lower bound: -27.1859429, upper bound: 27.1960934
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 2, lower bound: -27.1852035, upper bound: 27.1852029
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 2, lower bound: -27.1852035, upper bound: 27.1876654
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 2, lower bound: -27.1874860, upper bound: 27.1864186
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 2, lower bound: -27.1874860, upper bound: 27.1976968

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -16.8317299, 12.7335157, -16.8317299, 12.7335157, -29.5652428, 29.5652428
1: -13.4129200, 11.3248215, -13.4129200, 11.3248215, -24.7377396, 24.7377415
2: -22.4660797, 6.9903717, -22.4660797, 6.9903717, -29.4564514, 29.4564514
3: -19.9012680, 8.9243526, -19.9012680, 8.9243526, -28.8256187, 28.8256149
4: -20.2617760, 12.1930265, -20.2617760, 12.1930265, -32.4548035, 32.4548035
5: -14.9546680, 12.9803066, -14.9546680, 12.9803066, -27.9349747, 27.9349747
6: -16.0028496, 13.6762362, -16.0028496, 13.6762362, -29.6790848, 29.6790810
7: -18.0861378, 13.0791740, -18.0861378, 13.0791740, -31.1652985, 31.1653023
8: -20.8303032, 11.8382320, -20.8303032, 11.8382320, -32.6685333, 32.6685333
9: -14.4516153, 17.0915165, -14.4516153, 17.0915165, -31.5431328, 31.5431328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=246, inp2_unstable=246, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1723257, upper bound: 27.1717367
time: 3.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1697128, upper bound: 27.1697128
time: 15.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -16.8317299, 12.7335157, -17.3238983, 13.1115246, -29.9432545, 30.0574112
1: -13.4129200, 11.3248215, -13.8358364, 11.6765242, -25.0894432, 25.1606579
2: -22.4660797, 6.9903717, -23.0704117, 7.3563652, -29.8224449, 30.0607796
3: -19.9012680, 8.9243526, -20.5380898, 9.2507601, -29.1520195, 29.4624424
4: -20.2617760, 12.1930265, -20.7957764, 12.5690279, -32.8308029, 32.9888039
5: -14.9546680, 12.9803066, -15.4290638, 13.3535156, -28.3081818, 28.4093628
6: -16.0028496, 13.6762362, -16.4827061, 14.0766678, -30.0795135, 30.1589432
7: -18.0861378, 13.0791740, -18.6212692, 13.5060740, -31.5922012, 31.7004433
8: -20.8303032, 11.8382320, -21.4277573, 12.2034607, -33.0337639, 33.2659912
9: -14.4516153, 17.0915165, -14.8881588, 17.5661545, -32.0177689, 31.9796753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=89, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=14, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=246, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1723257, upper bound: 27.1729935
time: 3.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1697128, upper bound: 27.1716974
time: 8.16 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -17.3379898, 13.1264954, -16.8317299, 12.7335157, -30.0714989, 29.9582253
1: -13.8477888, 11.6870747, -13.4129200, 11.3248215, -25.1726112, 25.0999908
2: -23.0851631, 7.3696795, -22.4660797, 6.9903717, -30.0755348, 29.8357582
3: -20.5558205, 9.2606688, -19.9012680, 8.9243526, -29.4801731, 29.1619301
4: -20.8129292, 12.5806046, -20.2617760, 12.1930265, -33.0059547, 32.8423805
5: -15.4483376, 13.3657217, -14.9546680, 12.9803066, -28.4286366, 28.3203869
6: -16.4954891, 14.0895319, -16.0028496, 13.6762362, -30.1717186, 30.0923767
7: -18.6347389, 13.5205002, -18.0861378, 13.0791740, -31.7139130, 31.6066360
8: -21.4465790, 12.2142963, -20.8303032, 11.8382320, -33.2848129, 33.0445976
9: -14.9008160, 17.5803413, -14.4516153, 17.0915165, -31.9923325, 32.0319557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=246, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1729935, upper bound: 27.1742989
time: 7.19 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1716974, upper bound: 27.1727811
time: 7.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -17.3379898, 13.1264954, -17.3379898, 13.1264954, -30.4644852, 30.4644852
1: -13.8477888, 11.6870747, -13.8477888, 11.6870747, -25.5348625, 25.5348625
2: -23.0851631, 7.3696795, -23.0851631, 7.3696795, -30.4548397, 30.4548416
3: -20.5558205, 9.2606688, -20.5558205, 9.2606688, -29.8164902, 29.8164902
4: -20.8129292, 12.5806046, -20.8129292, 12.5806046, -33.3935318, 33.3935318
5: -15.4483376, 13.3657217, -15.4483376, 13.3657217, -28.8140526, 28.8140526
6: -16.4954891, 14.0895319, -16.4954891, 14.0895319, -30.5850182, 30.5850163
7: -18.6347389, 13.5205002, -18.6347389, 13.5205002, -32.1552391, 32.1552391
8: -21.4465790, 12.2142963, -21.4465790, 12.2142963, -33.6608734, 33.6608734
9: -14.9008160, 17.5803413, -14.9008160, 17.5803413, -32.4811516, 32.4811554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=89, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752784, upper bound: 27.1878163
time: 8.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1716974, upper bound: 27.1866209
time: 12.65 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -16.8317299, 12.7335157, -17.8874741, 13.5268631, -30.3585873, 30.6209888
1: -13.4129200, 11.3248215, -14.2625885, 12.0275040, -25.4404182, 25.5874081
2: -22.4660797, 6.9903717, -23.8239956, 7.5214262, -29.9875050, 30.8143673
3: -19.9012680, 8.9243526, -21.1708260, 9.4986572, -29.3999214, 30.0951786
4: -20.2617760, 12.1930265, -21.4935303, 12.9550762, -33.2168465, 33.6865578
5: -14.9546680, 12.9803066, -15.9107056, 13.7824078, -28.7370758, 28.8910084
6: -16.0028496, 13.6762362, -17.0131073, 14.5267038, -30.5295448, 30.6893387
7: -18.0861378, 13.0791740, -19.2299156, 13.9158087, -32.0019417, 32.3090820
8: -20.8303032, 11.8382320, -22.1316624, 12.5717964, -33.4020996, 33.9698944
9: -14.4516153, 17.0915165, -15.3638668, 18.1342430, -32.5858574, 32.4553833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=246, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1724571, upper bound: 27.1718093
time: 3.62 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1697928, upper bound: 27.1697680
time: 3.91 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -17.3238983, 13.1115246, -17.8874741, 13.5268631, -30.8507519, 30.9989986
1: -13.8358364, 11.6765242, -14.2625885, 12.0275040, -25.8633385, 25.9391041
2: -23.0704117, 7.3563652, -23.8239956, 7.5214262, -30.5918331, 31.1803608
3: -20.5380898, 9.2507601, -21.1708260, 9.4986572, -30.0367432, 30.4215832
4: -20.7957764, 12.5690279, -21.4935303, 12.9550762, -33.7508507, 34.0625572
5: -15.4290638, 13.3535156, -15.9107056, 13.7824078, -29.2114677, 29.2642155
6: -16.4827061, 14.0766678, -17.0131073, 14.5267038, -31.0094109, 31.0897675
7: -18.6212692, 13.5060740, -19.2299156, 13.9158087, -32.5370789, 32.7359848
8: -21.4277573, 12.2034607, -22.1316624, 12.5717964, -33.9995537, 34.3351212
9: -14.8881588, 17.5661545, -15.3638668, 18.1342430, -33.0223999, 32.9300232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1718375, upper bound: 27.1743074
time: 6.64 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1697928, upper bound: 27.1727760
time: 4.03 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -16.8317299, 12.7335157, -18.7389374, 14.2121658, -31.0438957, 31.4724541
1: -13.4129200, 11.3248215, -14.9956312, 12.6541805, -26.0671005, 26.3204536
2: -22.4660797, 6.9903717, -24.8272076, 8.2060719, -30.6721516, 31.8175793
3: -19.9012680, 8.9243526, -22.2329102, 10.0517826, -29.9530487, 31.1572609
4: -20.2617760, 12.1930265, -22.4428482, 13.6119614, -33.8737373, 34.6358757
5: -14.9546680, 12.9803066, -16.7553368, 14.4334555, -29.3881226, 29.7356434
6: -16.0028496, 13.6762362, -17.8754063, 15.2532148, -31.2560616, 31.5516434
7: -18.0861378, 13.0791740, -20.1374226, 14.6879034, -32.7740364, 33.2165985
8: -20.8303032, 11.8382320, -23.2031250, 13.2296448, -34.0599480, 35.0413589
9: -14.4516153, 17.0915165, -16.1392937, 18.9533577, -33.4049721, 33.2308121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=23, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=246, inp2_unstable=251, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1724571, upper bound: 27.1732254
time: 6.67 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1697928, upper bound: 27.1719930
time: 4.15 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -17.3379898, 13.1264954, -18.7389374, 14.2121658, -31.5501556, 31.8654327
1: -13.8477888, 11.6870747, -14.9956312, 12.6541805, -26.5019684, 26.6827049
2: -23.0851631, 7.3696795, -24.8272076, 8.2060719, -31.2912350, 32.1968842
3: -20.5558205, 9.2606688, -22.2329102, 10.0517826, -30.6076031, 31.4935760
4: -20.8129292, 12.5806046, -22.4428482, 13.6119614, -34.4248886, 35.0234528
5: -15.4483376, 13.3657217, -16.7553368, 14.4334555, -29.8817902, 30.1210594
6: -16.4954891, 14.0895319, -17.8754063, 15.2532148, -31.7486992, 31.9649353
7: -18.6347389, 13.5205002, -20.1374226, 14.6879034, -33.3226433, 33.6579208
8: -21.4465790, 12.2142963, -23.2031250, 13.2296448, -34.6762238, 35.4174156
9: -14.9008160, 17.5803413, -16.1392937, 18.9533577, -33.8541718, 33.7196312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=23, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=251, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1724571, upper bound: 27.1880622
time: 6.84 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1697928, upper bound: 27.1870516
time: 6.03 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -17.8874741, 13.5268631, -16.8317299, 12.7335157, -30.6209908, 30.3585892
1: -14.2625885, 12.0275040, -13.4129200, 11.3248215, -25.5874062, 25.4404202
2: -23.8239956, 7.5214262, -22.4660797, 6.9903717, -30.8143673, 29.9875031
3: -21.1708260, 9.4986572, -19.9012680, 8.9243526, -30.0951786, 29.3999176
4: -21.4935303, 12.9550762, -20.2617760, 12.1930265, -33.6865578, 33.2168503
5: -15.9107056, 13.7824078, -14.9546680, 12.9803066, -28.8910046, 28.7370758
6: -17.0131073, 14.5267038, -16.0028496, 13.6762362, -30.6893425, 30.5295486
7: -19.2299156, 13.9158087, -18.0861378, 13.0791740, -32.3090820, 32.0019455
8: -22.1316624, 12.5717964, -20.8303032, 11.8382320, -33.9698944, 33.4020996
9: -15.3638668, 18.1342430, -14.4516153, 17.0915165, -32.4553833, 32.5858574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=91, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=246, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1718093, upper bound: 27.1724571
time: 4.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1697680, upper bound: 27.1697928
time: 3.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -17.8874741, 13.5268631, -17.3238983, 13.1115246, -30.9989986, 30.8507538
1: -14.2625885, 12.0275040, -13.8358364, 11.6765242, -25.9391060, 25.8633404
2: -23.8239956, 7.5214262, -23.0704117, 7.3563652, -31.1803608, 30.5918350
3: -21.1708260, 9.4986572, -20.5380898, 9.2507601, -30.4215813, 30.0367470
4: -21.4935303, 12.9550762, -20.7957764, 12.5690279, -34.0625572, 33.7508545
5: -15.9107056, 13.7824078, -15.4290638, 13.3535156, -29.2642174, 29.2114716
6: -17.0131073, 14.5267038, -16.4827061, 14.0766678, -31.0897675, 31.0094109
7: -19.2299156, 13.9158087, -18.6212692, 13.5060740, -32.7359848, 32.5370789
8: -22.1316624, 12.5717964, -21.4277573, 12.2034607, -34.3351212, 33.9995537
9: -15.3638668, 18.1342430, -14.8881588, 17.5661545, -32.9300232, 33.0223999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=91, inp2_unstable=89, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1723577, upper bound: 27.1718374
time: 13.78 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1697680, upper bound: 27.1717263
time: 6.05 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -18.7389374, 14.2121658, -16.8317299, 12.7335157, -31.4724541, 31.0438957
1: -14.9956312, 12.6541805, -13.4129200, 11.3248215, -26.3204536, 26.0671005
2: -24.8272076, 8.2060719, -22.4660797, 6.9903717, -31.8175774, 30.6721516
3: -22.2329102, 10.0517826, -19.9012680, 8.9243526, -31.1572609, 29.9530487
4: -22.4428482, 13.6119614, -20.2617760, 12.1930265, -34.6358757, 33.8737373
5: -16.7553368, 14.4334555, -14.9546680, 12.9803066, -29.7356396, 29.3881226
6: -17.8754063, 15.2532148, -16.0028496, 13.6762362, -31.5516434, 31.2560654
7: -20.1374226, 14.6879034, -18.0861378, 13.0791740, -33.2165985, 32.7740402
8: -23.2031250, 13.2296448, -20.8303032, 11.8382320, -35.0413589, 34.0599480
9: -16.1392937, 18.9533577, -14.4516153, 17.0915165, -33.2308121, 33.4049721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=246, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1732254, upper bound: 27.1723257
time: 15.26 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1719930, upper bound: 27.1732357
time: 11.08 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -18.7389374, 14.2121658, -17.3379898, 13.1264954, -31.8654327, 31.5501556
1: -14.9956312, 12.6541805, -13.8477888, 11.6870747, -26.6827011, 26.5019684
2: -24.8272076, 8.2060719, -23.0851631, 7.3696795, -32.1968842, 31.2912350
3: -22.2329102, 10.0517826, -20.5558205, 9.2606688, -31.4935780, 30.6076031
4: -22.4428482, 13.6119614, -20.8129292, 12.5806046, -35.0234528, 34.4248886
5: -16.7553368, 14.4334555, -15.4483376, 13.3657217, -30.1210556, 29.8817902
6: -17.8754063, 15.2532148, -16.4954891, 14.0895319, -31.9649353, 31.7487011
7: -20.1374226, 14.6879034, -18.6347389, 13.5205002, -33.6579208, 33.3226433
8: -23.2031250, 13.2296448, -21.4465790, 12.2142963, -35.4174156, 34.6762238
9: -16.1392937, 18.9533577, -14.9008160, 17.5803413, -33.7196312, 33.8541679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=89, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=14, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1732254, upper bound: 27.1887686
time: 8.29 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1719930, upper bound: 27.1873746
time: 32.50 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -17.8874741, 13.5268631, -17.8874741, 13.5268631, -31.4143353, 31.4143372
1: -14.2625885, 12.0275040, -14.2625885, 12.0275040, -26.2900925, 26.2900867
2: -23.8239956, 7.5214262, -23.8239956, 7.5214262, -31.3454208, 31.3454208
3: -21.1708260, 9.4986572, -21.1708260, 9.4986572, -30.6694832, 30.6694813
4: -21.4935303, 12.9550762, -21.4935303, 12.9550762, -34.4486046, 34.4486084
5: -15.9107056, 13.7824078, -15.9107056, 13.7824078, -29.6931114, 29.6931133
6: -17.0131073, 14.5267038, -17.0131073, 14.5267038, -31.5398064, 31.5398026
7: -19.2299156, 13.9158087, -19.2299156, 13.9158087, -33.1457253, 33.1457214
8: -22.1316624, 12.5717964, -22.1316624, 12.5717964, -34.7034607, 34.7034607
9: -15.3638668, 18.1342430, -15.3638668, 18.1342430, -33.4981079, 33.4981079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=91, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1612729, upper bound: 27.1586973
time: 6.37 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1555969, upper bound: 27.1555969
time: 2.32 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -18.7153397, 14.1912203, -17.8874741, 13.5268631, -32.2422028, 32.0786934
1: -14.9776440, 12.6317577, -14.2625885, 12.0275040, -27.0051479, 26.8943443
2: -24.8011169, 8.1786776, -23.8239956, 7.5214262, -32.3225403, 32.0026703
3: -22.1939545, 10.0299835, -21.1708260, 9.4986572, -31.6926003, 31.2008076
4: -22.4154968, 13.5932827, -21.4935303, 12.9550762, -35.3705750, 35.0868111
5: -16.7348785, 14.4133339, -15.9107056, 13.7824078, -30.5172863, 30.3240299
6: -17.8531113, 15.2300110, -17.0131073, 14.5267038, -32.3798141, 32.2431183
7: -20.1147156, 14.6646862, -19.2299156, 13.9158087, -34.0305252, 33.8945999
8: -23.1734371, 13.2032824, -22.1316624, 12.5717964, -35.7452316, 35.3349457
9: -16.1142273, 18.9304619, -15.3638668, 18.1342430, -34.2484703, 34.2943268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1788006, upper bound: 27.1812706
time: 4.22 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1771024, upper bound: 27.1799068
time: 14.52 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -17.8874741, 13.5268631, -18.7389374, 14.2121658, -32.0996399, 32.2658005
1: -14.2625885, 12.0275040, -14.9956312, 12.6541805, -26.9167671, 27.0231323
2: -23.8239956, 7.5214262, -24.8272076, 8.2060719, -32.0300674, 32.3486328
3: -21.1708260, 9.4986572, -22.2329102, 10.0517826, -31.2226086, 31.7315636
4: -21.4935303, 12.9550762, -22.4428482, 13.6119614, -35.1054916, 35.3979263
5: -15.9107056, 13.7824078, -16.7553368, 14.4334555, -30.3441582, 30.5377445
6: -17.0131073, 14.5267038, -17.8754063, 15.2532148, -32.2663193, 32.4021072
7: -19.2299156, 13.9158087, -20.1374226, 14.6879034, -33.9178123, 34.0532303
8: -22.1316624, 12.5717964, -23.2031250, 13.2296448, -35.3613052, 35.7749214
9: -15.3638668, 18.1342430, -16.1392937, 18.9533577, -34.3172226, 34.2735291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=91, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=23, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=251, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1784344, upper bound: 27.1806819
time: 9.39 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1771024, upper bound: 27.1787451
time: 6.11 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -18.7389374, 14.2121658, -18.7389374, 14.2121658, -32.9511032, 32.9511032
1: -14.9956312, 12.6541805, -14.9956312, 12.6541805, -27.6498108, 27.6498108
2: -24.8272076, 8.2060719, -24.8272076, 8.2060719, -33.0332794, 33.0332794
3: -22.2329102, 10.0517826, -22.2329102, 10.0517826, -32.2846909, 32.2846909
4: -22.4428482, 13.6119614, -22.4428482, 13.6119614, -36.0548096, 36.0548096
5: -16.7553368, 14.4334555, -16.7553368, 14.4334555, -31.1887932, 31.1887932
6: -17.8754063, 15.2532148, -17.8754063, 15.2532148, -33.1286201, 33.1286201
7: -20.1374226, 14.6879034, -20.1374226, 14.6879034, -34.8253250, 34.8253250
8: -23.2031250, 13.2296448, -23.2031250, 13.2296448, -36.4327660, 36.4327698
9: -16.1392937, 18.9533577, -16.1392937, 18.9533577, -35.0926437, 35.0926437

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1587670, upper bound: 27.1843221
time: 8.26 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1555969, upper bound: 27.1555969
time: 25.36 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 43.74 seconds
IS_A1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1723257, upper bound: 27.1717367
IS_A1_B1_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1697128, upper bound: 27.1697128
IS_A1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1723257, upper bound: 27.1729935
IS_A1_B1_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1697128, upper bound: 27.1716974
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1729935, upper bound: 27.1742989
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1716974, upper bound: 27.1727811
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1752784, upper bound: 27.1878163
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1716974, upper bound: 27.1866209
IS_A1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1724571, upper bound: 27.1718093
IS_A1_B2_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1697928, upper bound: 27.1697680
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1718375, upper bound: 27.1743074
IS_A1_B2_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1697928, upper bound: 27.1727760
IS_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1724571, upper bound: 27.1732254
IS_A1_B2_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1697928, upper bound: 27.1719930
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1724571, upper bound: 27.1880622
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1697928, upper bound: 27.1870516
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1718093, upper bound: 27.1724571
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1697680, upper bound: 27.1697928
IS_A2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1723577, upper bound: 27.1718374
IS_A2_B1_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1697680, upper bound: 27.1717263
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1732254, upper bound: 27.1723257
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1719930, upper bound: 27.1732357
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1732254, upper bound: 27.1887686
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1719930, upper bound: 27.1873746
IS_A2_B2_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1612729, upper bound: 27.1586973
IS_A2_B2_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1555969, upper bound: 27.1555969
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1788006, upper bound: 27.1812706
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1771024, upper bound: 27.1799068
IS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1784344, upper bound: 27.1806819
IS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1771024, upper bound: 27.1787451
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1587670, upper bound: 27.1843221
IS_A2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 43.74
Output dim: 2, lower bound: -27.1555969, upper bound: 27.1555969

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -16.8125973, 12.7328911, -16.8016129, 12.7111797, -29.5237770, 29.5345001
1: -13.4218760, 11.3385963, -13.3884916, 11.3049088, -24.7267818, 24.7270889
2: -22.3968735, 7.1381764, -22.4261818, 6.9775701, -29.3744392, 29.5643578
3: -19.9246082, 8.9845314, -19.8650570, 8.9087105, -28.8333187, 28.8495884
4: -20.1898422, 12.2064762, -20.2259445, 12.1716232, -32.3614655, 32.4324188
5: -14.9760876, 12.9663172, -14.9279737, 12.9573383, -27.9334221, 27.8942909
6: -15.9941921, 13.6665220, -15.9741259, 13.6519890, -29.6461811, 29.6406479
7: -18.0708981, 13.1129217, -18.0536880, 13.0559940, -31.1268921, 31.1666069
8: -20.7974510, 11.8515377, -20.7930851, 11.8175049, -32.6149559, 32.6446228
9: -14.4500961, 17.0535355, -14.4258394, 17.0612526, -31.5113487, 31.4793739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=246, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1716974, upper bound: 27.1727811
time: 5.28 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1716974, upper bound: 27.1727811
time: 4.07 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -17.3072186, 13.1034393, -16.8125973, 12.7328911, -30.0401077, 29.9160366
1: -13.8228397, 11.6666660, -13.4218760, 11.3385963, -25.1614323, 25.0885353
2: -23.0448456, 7.3561068, -22.3968735, 7.1381764, -30.1830215, 29.7529736
3: -20.5188541, 9.2444859, -19.9246082, 8.9845314, -29.5033798, 29.1690941
4: -20.7764225, 12.5586872, -20.1898422, 12.2064762, -32.9828987, 32.7485275
5: -15.4206753, 13.3423338, -14.9760876, 12.9663172, -28.3869934, 28.3184204
6: -16.4661255, 14.0647621, -15.9941921, 13.6665220, -30.1326485, 30.0589542
7: -18.6017113, 13.4966183, -18.0708981, 13.1129217, -31.7146339, 31.5675144
8: -21.4085598, 12.1930447, -20.7974510, 11.8515377, -33.2600975, 32.9904938
9: -14.8744125, 17.5494862, -14.4500961, 17.0535355, -31.9279480, 31.9995823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=89, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1866107, upper bound: 27.1866209
time: 8.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1866107, upper bound: 27.1866209
time: 4.97 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 15.27 seconds
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 15.27
Output dim: 2, lower bound: -27.1716974, upper bound: 27.1727811
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 15.27
Output dim: 2, lower bound: -27.1716974, upper bound: 27.1727811
IS_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 15.27
Output dim: 2, lower bound: -27.1866107, upper bound: 27.1866209
IS_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 15.27
Output dim: 2, lower bound: -27.1866107, upper bound: 27.1866209
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 15.27
Output dim: 2, lower bound: -27.1716974, upper bound: 27.1866209
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.27
Output dim: 2, lower bound: -27.1718375, upper bound: 27.1743074
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 15.27
Output dim: 2, lower bound: -27.1724571, upper bound: 27.1880622
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 15.27
Output dim: 2, lower bound: -27.1697928, upper bound: 27.1870516
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.27
Output dim: 2, lower bound: -27.1732254, upper bound: 27.1887686
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.27
Output dim: 2, lower bound: -27.1719930, upper bound: 27.1873746
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 15.27
Output dim: 2, lower bound: -27.1788006, upper bound: 27.1812706
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 15.27
Output dim: 2, lower bound: -27.1771024, upper bound: 27.1799068
IS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 15.27
Output dim: 2, lower bound: -27.1784344, upper bound: 27.1806819
IS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 15.27
Output dim: 2, lower bound: -27.1771024, upper bound: 27.1787451
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.27
Output dim: 2, lower bound: -27.1587670, upper bound: 27.1843221

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 7.89 + 594.01 = 601.91 seconds
