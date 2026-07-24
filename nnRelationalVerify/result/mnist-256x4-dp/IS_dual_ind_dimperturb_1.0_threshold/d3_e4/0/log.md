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
execution time: IAR + RelationalAnalysis = 1.53 + 6.41 = 7.94 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -27.2005054, upper bound: 27.2005054

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1972990, upper bound: 27.1969081
time: 5.54 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987716, upper bound: 27.1987716
time: 5.21 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.89 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.89
Output dim: 2, lower bound: -27.1972990, upper bound: 27.1969081
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.89
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

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1972990, upper bound: 27.1969037
time: 3.94 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1972990, upper bound: 27.1969065
time: 4.79 seconds

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

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987596, upper bound: 27.1987593
time: 8.11 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987716, upper bound: 27.1987716
time: 4.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 14.54 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.54
Output dim: 2, lower bound: -27.1972990, upper bound: 27.1969037
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.54
Output dim: 2, lower bound: -27.1972990, upper bound: 27.1969065
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.54
Output dim: 2, lower bound: -27.1987596, upper bound: 27.1987593
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.54
Output dim: 2, lower bound: -27.1987716, upper bound: 27.1987716

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -17.3370628, 13.1330891, -17.6276932, 13.3823795, -30.7194405, 30.7607822
1: -13.8530216, 11.6943493, -14.1119061, 11.9265022, -25.7795238, 25.8062515
2: -23.0674133, 7.4033875, -23.3507347, 7.7548943, -30.8223076, 30.7541199
3: -20.5658150, 9.2729082, -20.9079361, 9.4832172, -30.0490322, 30.1808434
4: -20.8094063, 12.5843725, -21.1190586, 12.8308964, -33.6403046, 33.7034302
5: -15.4613705, 13.3685026, -15.7731667, 13.5933065, -29.0546761, 29.1416702
6: -16.4964905, 14.0960989, -16.8093491, 14.3677006, -30.8641891, 30.9054413
7: -18.6369781, 13.5307159, -18.9578648, 13.8396053, -32.4765854, 32.4885788
8: -21.4527378, 12.2158527, -21.8363914, 12.4509439, -33.9036789, 34.0522461
9: -14.9058094, 17.5759335, -15.1942959, 17.8303185, -32.7361298, 32.7702255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=26, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=251, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1922682, upper bound: 27.1924932
time: 5.79 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1914562, upper bound: 27.1910880
time: 15.29 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -17.5927582, 13.3273621, -19.0028114, 14.4149561, -32.0077057, 32.3301735
1: -14.0613518, 11.8653307, -15.2250328, 12.8454943, -26.9068451, 27.0903625
2: -23.3992157, 7.5290685, -25.1455765, 8.3917503, -31.7909660, 32.6746445
3: -20.8711929, 9.4107084, -22.5602036, 10.2133112, -31.0845032, 31.9709110
4: -21.1102295, 12.7700186, -22.7534103, 13.8232403, -34.9334717, 35.5234299
5: -15.6937466, 13.5625000, -17.0077801, 14.6391811, -30.3329277, 30.5702782
6: -16.7422428, 14.3019047, -18.1344185, 15.4804792, -32.2227211, 32.4363251
7: -18.9111023, 13.7342377, -20.4248276, 14.9128685, -33.8239708, 34.1590652
8: -21.7673512, 12.3959579, -23.5417595, 13.4206448, -35.1879959, 35.9377174
9: -15.1254759, 17.8310757, -16.3767147, 19.2071800, -34.3326569, 34.2077904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=28, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=252, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1922428, upper bound: 27.1924823
time: 6.70 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1914297, upper bound: 27.1910797
time: 5.41 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -18.7854786, 14.2485313, -18.0481148, 13.7090950, -32.4945679, 32.2966423
1: -15.0435600, 12.6955757, -14.4594097, 12.2307062, -27.2742653, 27.1549854
2: -24.8579922, 8.2737198, -23.8371716, 8.0769901, -32.9349747, 32.1108856
3: -22.3003845, 10.0943928, -21.4192886, 9.7487450, -32.0491295, 31.5136795
4: -22.4895763, 13.6540966, -21.5930634, 13.1559553, -35.6455307, 35.2471619
5: -16.8097725, 14.4704704, -16.1716652, 13.9128895, -30.7226620, 30.6421280
6: -17.9238014, 15.3011847, -17.2357769, 14.7252369, -32.6490364, 32.5369606
7: -20.1883736, 14.7397346, -19.4009266, 14.2143097, -34.4026833, 34.1406555
8: -23.2649822, 13.2635117, -22.3658028, 12.7738342, -36.0388107, 35.6293106
9: -16.1866035, 18.9885826, -15.5786438, 18.2212715, -34.4078751, 34.5672226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=30, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=252, inp2_unstable=253, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1983060, upper bound: 27.1982240
time: 5.49 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1986913, upper bound: 27.1986800
time: 8.26 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -19.0930195, 14.4850512, -19.4342861, 14.7581120, -33.8511276, 33.9193306
1: -15.2954569, 12.9083900, -15.5850296, 13.1613369, -28.4567947, 28.4934196
2: -25.2478142, 8.4515581, -25.6462631, 8.7288589, -33.9766693, 34.0978165
3: -22.6728954, 10.2697258, -23.0882168, 10.4903326, -33.1632271, 33.3579407
4: -22.8479252, 13.8824434, -23.2434616, 14.1596613, -37.0075874, 37.1259003
5: -17.0913181, 14.7049208, -17.4215775, 14.9717159, -32.0630341, 32.1264992
6: -18.2270794, 15.5543003, -18.5728054, 15.8502941, -34.0773697, 34.1271057
7: -20.5140228, 14.9919472, -20.8795033, 15.3032761, -35.8172989, 35.8714485
8: -23.6493340, 13.4916534, -24.0835838, 13.7568226, -37.4061584, 37.5752373
9: -16.4574127, 19.2913055, -16.7756424, 19.6099205, -36.0673294, 36.0669479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=33, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=252, inp2_unstable=253, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1983376, upper bound: 27.1982713
time: 13.98 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987290, upper bound: 27.1987290
time: 5.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.32 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 2, lower bound: -27.1922682, upper bound: 27.1924932
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 2, lower bound: -27.1914562, upper bound: 27.1910880
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 2, lower bound: -27.1922428, upper bound: 27.1924823
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 2, lower bound: -27.1914297, upper bound: 27.1910797
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 2, lower bound: -27.1983060, upper bound: 27.1982240
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 2, lower bound: -27.1986913, upper bound: 27.1986800
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 2, lower bound: -27.1983376, upper bound: 27.1982713
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 2, lower bound: -27.1987290, upper bound: 27.1987290

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -16.7611084, 12.7015238, -17.4635105, 13.2587433, -30.0198517, 30.1650352
1: -13.3855009, 11.3122377, -13.9779024, 11.8166933, -25.2021942, 25.2901402
2: -22.3123016, 7.1470265, -23.1363297, 7.6785393, -29.9908409, 30.2833557
3: -19.8758621, 8.9715996, -20.7107811, 9.3963175, -29.2721786, 29.6823807
4: -20.1252022, 12.1731739, -20.9243851, 12.7126522, -32.8378525, 33.0975533
5: -14.9436359, 12.9313087, -15.6247940, 13.4686680, -28.4123039, 28.5561028
6: -15.9472513, 13.6328230, -16.6513863, 14.2345819, -30.1818275, 30.2842102
7: -18.0196018, 13.0838747, -18.7823830, 13.7111149, -31.7307167, 31.8662567
8: -20.7419281, 11.8177338, -21.6327553, 12.3358212, -33.0777512, 33.4504890
9: -14.4135876, 16.9984055, -15.0529184, 17.6658764, -32.0794601, 32.0513229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=26, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=250, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1920667, upper bound: 27.1923440
time: 6.81 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1920848, upper bound: 27.1923600
time: 5.04 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -19.7926025, 15.0098991, -17.1615486, 13.0319195, -32.8245239, 32.1714478
1: -15.8650379, 13.3626194, -13.7317009, 11.6145725, -27.4796104, 27.0943203
2: -26.1953201, 8.6536350, -22.7421150, 7.5410781, -33.7363930, 31.3957500
3: -23.5491657, 10.6173306, -20.3493538, 9.2377615, -32.7869186, 30.9666843
4: -23.7330208, 14.3734350, -20.5664139, 12.4954166, -36.2284393, 34.9398499
5: -17.7322464, 15.2470036, -15.3522120, 13.2394705, -30.9717178, 30.5992050
6: -18.8550873, 16.1071377, -16.3611298, 13.9896927, -32.8447762, 32.4682693
7: -21.2829590, 15.5005894, -18.4591484, 13.4756565, -34.7586136, 33.9597359
8: -24.5181770, 13.9240952, -21.2576618, 12.1251736, -36.6433487, 35.1817513
9: -17.0485058, 20.0201340, -14.7933674, 17.3633423, -34.4118385, 34.8134995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=91, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=253, inp2_unstable=250, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912846, upper bound: 27.1909368
time: 4.77 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912946, upper bound: 27.1909477
time: 6.57 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -17.0233688, 12.9005690, -18.8389835, 14.2913227, -31.3146915, 31.7395496
1: -13.5988941, 11.4872417, -15.0911036, 12.7355528, -26.3344460, 26.5783463
2: -22.6535301, 7.2747264, -24.9321651, 8.3147659, -30.9682961, 32.2068901
3: -20.1892662, 9.1124811, -22.3620949, 10.1260185, -30.3152847, 31.4745750
4: -20.4339523, 12.3632784, -22.5590744, 13.7051744, -34.1391258, 34.9223518
5: -15.1817360, 13.1301622, -16.8597755, 14.5146275, -29.6963615, 29.9899368
6: -16.1990471, 13.8436117, -17.9767685, 15.3472919, -31.5463390, 31.8203754
7: -18.3009148, 13.2923727, -20.2498150, 14.7846193, -33.0855331, 33.5421829
8: -21.0644188, 12.0019073, -23.3385391, 13.3053226, -34.3697281, 35.3404465
9: -14.6387815, 17.2602444, -16.2352409, 19.0429306, -33.6817131, 33.4954834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=28, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=252, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1920447, upper bound: 27.1923309
time: 5.19 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1920548, upper bound: 27.1923435
time: 6.15 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -20.0843010, 15.2357140, -18.5362091, 14.0626316, -34.1469345, 33.7719231
1: -16.1000633, 13.5616837, -14.8433132, 12.5322275, -28.6322899, 28.4049950
2: -26.5721188, 8.8119392, -24.5363045, 8.1734180, -34.7455330, 33.3482437
3: -23.8904972, 10.7773151, -21.9966831, 9.9649811, -33.8554764, 32.7739944
4: -24.0719891, 14.5917912, -22.1993427, 13.4865837, -37.5585709, 36.7911339
5: -17.9997292, 15.4680061, -16.5862865, 14.2837315, -32.2834549, 32.0542908
6: -19.1487217, 16.3430138, -17.6854897, 15.1008635, -34.2495804, 34.0285034
7: -21.5906334, 15.7333527, -19.9259377, 14.5476456, -36.1382751, 35.6592827
8: -24.8916893, 14.1434441, -22.9623375, 13.0920734, -37.9837646, 37.1057816
9: -17.3014793, 20.3118572, -15.9739494, 18.7387543, -36.0402298, 36.2858047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=92, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=28, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=253, inp2_unstable=252, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912625, upper bound: 27.1909302
time: 7.31 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912740, upper bound: 27.1909400
time: 6.03 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -16.9454861, 12.8668261, -17.3988724, 13.2133541, -30.1588402, 30.2656975
1: -13.5515566, 11.4587059, -13.9286137, 11.7878408, -25.3393974, 25.3873196
2: -22.4708424, 7.4094872, -22.9998798, 7.7392931, -30.2101326, 30.4093590
3: -20.1069412, 9.1206455, -20.6440544, 9.3951778, -29.5021191, 29.7646999
4: -20.3167496, 12.3378563, -20.8253002, 12.6819410, -32.9986877, 33.1631546
5: -15.1531582, 13.0687008, -15.5798740, 13.4151173, -28.5682716, 28.6485710
6: -16.1405029, 13.8068848, -16.5962906, 14.1920795, -30.3325825, 30.4031715
7: -18.2180462, 13.2977200, -18.7080917, 13.6946821, -31.9127274, 32.0058136
8: -20.9678802, 11.9746151, -21.5462704, 12.3052216, -33.2731018, 33.5208855
9: -14.5976534, 17.1575851, -15.0114813, 17.5755882, -32.1732407, 32.1690674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=27, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=252, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1941215, upper bound: 27.1935276
time: 4.39 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927327, upper bound: 27.1927070
time: 8.58 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -18.0727615, 13.7078123, -17.7779789, 13.5019989, -31.5747566, 31.4857903
1: -14.4662971, 12.2095051, -14.2388258, 12.0458632, -26.5121574, 26.4483299
2: -23.9460373, 7.9026222, -23.4901066, 7.9343195, -31.8803558, 31.3927288
3: -21.4527779, 9.7081995, -21.0959034, 9.6011181, -31.0538902, 30.8041000
4: -21.6499386, 13.1350212, -21.2730961, 12.9585218, -34.6084595, 34.4081192
5: -16.1629543, 13.9270782, -15.9248333, 13.7055550, -29.8685093, 29.8519058
6: -17.2218952, 14.7180271, -16.9685650, 14.5027180, -31.7246132, 31.6865921
7: -19.4276237, 14.1729460, -19.1124020, 13.9975815, -33.4252052, 33.2853432
8: -22.3653793, 12.7510519, -22.0237732, 12.5780878, -34.9434662, 34.7748184
9: -15.5645599, 18.2840977, -15.3417206, 17.9526691, -33.5172272, 33.6258163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=28, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=252, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1941593, upper bound: 27.1935760
time: 7.26 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1928051, upper bound: 27.1927796
time: 5.10 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -17.2527294, 13.0983181, -18.7897644, 14.2577639, -31.5104866, 31.8880806
1: -13.8006544, 11.6674900, -15.0551748, 12.7168999, -26.5175533, 26.7226639
2: -22.8606625, 7.5734234, -24.8164902, 8.3835373, -31.2441998, 32.3899117
3: -20.4755478, 9.2910318, -22.3137760, 10.1348743, -30.6104221, 31.6048088
4: -20.6735802, 12.5619392, -22.4773674, 13.6854219, -34.3590012, 35.0393066
5: -15.4310980, 13.3005295, -16.8287430, 14.4734278, -29.9045258, 30.1292725
6: -16.4368877, 14.0558109, -17.9373150, 15.3161306, -31.7530174, 31.9931240
7: -18.5442238, 13.5462551, -20.1908684, 14.7822590, -33.3264847, 33.7371216
8: -21.3451881, 12.1941366, -23.2707291, 13.2857018, -34.6308899, 35.4648590
9: -14.8643236, 17.4581394, -16.2072296, 18.9668121, -33.8311348, 33.6653557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=29, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=253, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1941304, upper bound: 27.1935494
time: 5.86 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927096, upper bound: 27.1927048
time: 4.20 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -18.3778687, 13.9387369, -19.1625786, 14.5461378, -32.9240074, 33.1013145
1: -14.7128162, 12.4189720, -15.3619604, 12.9737701, -27.6865864, 27.7809334
2: -24.3315048, 8.0744801, -25.2989197, 8.5815573, -32.9130630, 33.3733978
3: -21.8161564, 9.8798647, -22.7616711, 10.3397675, -32.1559219, 32.6415367
4: -22.0021095, 13.3599682, -22.9199524, 13.9593830, -35.9614944, 36.2799225
5: -16.4401760, 14.1572275, -17.1705513, 14.7608681, -31.2010422, 31.3277779
6: -17.5210114, 14.9654789, -18.3045349, 15.6245995, -33.1456070, 33.2700081
7: -19.7501793, 14.4206543, -20.5894966, 15.0829058, -34.8330841, 35.0101509
8: -22.7443638, 12.9743996, -23.7414589, 13.5576429, -36.3020058, 36.7158585
9: -15.8308773, 18.5807457, -16.5352421, 19.3390675, -35.1699371, 35.1159897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=253, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1941694, upper bound: 27.1935960
time: 6.89 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927763, upper bound: 27.1927763
time: 7.44 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 15.92 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.92
Output dim: 2, lower bound: -27.1920667, upper bound: 27.1923440
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.92
Output dim: 2, lower bound: -27.1920848, upper bound: 27.1923600
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.92
Output dim: 2, lower bound: -27.1912846, upper bound: 27.1909368
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.92
Output dim: 2, lower bound: -27.1912946, upper bound: 27.1909477
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.92
Output dim: 2, lower bound: -27.1920447, upper bound: 27.1923309
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.92
Output dim: 2, lower bound: -27.1920548, upper bound: 27.1923435
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.92
Output dim: 2, lower bound: -27.1912625, upper bound: 27.1909302
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.92
Output dim: 2, lower bound: -27.1912740, upper bound: 27.1909400
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.92
Output dim: 2, lower bound: -27.1941215, upper bound: 27.1935276
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.92
Output dim: 2, lower bound: -27.1927327, upper bound: 27.1927070
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.92
Output dim: 2, lower bound: -27.1941593, upper bound: 27.1935760
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.92
Output dim: 2, lower bound: -27.1928051, upper bound: 27.1927796
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.92
Output dim: 2, lower bound: -27.1941304, upper bound: 27.1935494
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.92
Output dim: 2, lower bound: -27.1927096, upper bound: 27.1927048
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.92
Output dim: 2, lower bound: -27.1941694, upper bound: 27.1935960
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.92
Output dim: 2, lower bound: -27.1927763, upper bound: 27.1927763

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -16.2179451, 12.2932873, -15.6298113, 11.8864088, -28.1043549, 27.9230938
1: -12.9455462, 10.9510746, -12.4897137, 10.5909882, -23.5365257, 23.4407864
2: -21.6050339, 6.8998470, -20.7523842, 6.8433084, -28.4483414, 27.6522312
3: -19.2290173, 8.6864014, -18.5304756, 8.4357395, -27.6647568, 27.2168770
4: -19.4810944, 11.7858524, -18.7578888, 11.4063644, -30.8874512, 30.5437412
5: -14.4570465, 12.5176792, -13.9771748, 12.0732813, -26.5303268, 26.4948540
6: -15.4295273, 13.1952209, -14.8856564, 12.7476292, -28.1771564, 28.0808773
7: -17.4357662, 12.6596889, -16.8171005, 12.2798710, -29.7156277, 29.4767876
8: -20.0704784, 11.4425259, -19.3510685, 11.0646057, -31.1350822, 30.7935944
9: -13.9471302, 16.4572392, -13.4754925, 15.8381977, -29.7853279, 29.9327316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=87, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=19, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=246, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1756639, upper bound: 27.1765012
time: 4.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1725830, upper bound: 27.1728371
time: 6.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -16.5418148, 12.5360937, -16.7633343, 12.7316513, -29.2734604, 29.2994232
1: -13.2079363, 11.1662560, -13.4110098, 11.3436203, -24.5515556, 24.5772667
2: -22.0281219, 7.0460815, -22.2385750, 7.3291588, -29.3572807, 29.2846508
3: -19.6156807, 8.8558521, -19.8834877, 9.0235682, -28.6392441, 28.7393341
4: -19.8647614, 12.0166845, -20.0989819, 12.2082129, -32.0729752, 32.1156654
5: -14.7464275, 12.7638063, -14.9917889, 12.9358768, -27.6823006, 27.7555885
6: -15.7379799, 13.4557514, -15.9685974, 13.6642838, -29.4022598, 29.4243488
7: -17.7834473, 12.9121981, -18.0321121, 13.1566267, -30.9400749, 30.9443092
8: -20.4699364, 11.6668720, -20.7552090, 11.8434219, -32.3133583, 32.4220810
9: -14.2250366, 16.7799129, -14.4455919, 16.9747581, -31.1997929, 31.2254982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=89, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=20, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=250, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1758434, upper bound: 27.1766583
time: 4.09 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1727857, upper bound: 27.1729945
time: 4.22 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -19.2450123, 14.5961208, -15.3295527, 11.6617451, -30.9067497, 29.9256744
1: -15.4198303, 12.9963255, -12.2444410, 10.3909225, -25.8107471, 25.2407627
2: -25.4823742, 8.4027948, -20.3610001, 6.7097507, -32.1921234, 28.7637939
3: -22.8974152, 10.3273478, -18.1707001, 8.2786274, -31.1760426, 28.4980450
4: -23.0817757, 13.9818602, -18.4011192, 11.1912985, -34.2730751, 32.3829727
5: -17.2387733, 14.8279772, -13.7066708, 11.8452501, -29.0840225, 28.5346489
6: -18.3318958, 15.6644859, -14.5992002, 12.5039988, -30.8358879, 30.2636833
7: -20.6928139, 15.0706997, -16.4953308, 12.0465832, -32.7393951, 31.5660305
8: -23.8382378, 13.5459690, -18.9794693, 10.8569651, -34.6951981, 32.5254326
9: -16.5768471, 19.4735603, -13.2177391, 15.5371761, -32.1140213, 32.6912994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=86, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=19, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=252, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1746570, upper bound: 27.1750444
time: 26.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1715719, upper bound: 27.1711670
time: 10.19 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -19.5698204, 14.8416986, -16.4579887, 12.5029869, -32.0728035, 31.2996826
1: -15.6843138, 13.2138824, -13.1619911, 11.1399097, -26.8242226, 26.3758698
2: -25.9062767, 8.5515594, -21.8390884, 7.1928687, -33.0991440, 30.3906479
3: -23.2841377, 10.4993858, -19.5180187, 8.8640156, -32.1481552, 30.0174046
4: -23.4680939, 14.2142210, -19.7367096, 11.9895000, -35.4575958, 33.9509239
5: -17.5315762, 15.0766478, -14.7168732, 12.7039700, -30.2355423, 29.7935219
6: -18.6423893, 15.9270267, -15.6771812, 13.4170904, -32.0594788, 31.6042042
7: -21.0427799, 15.3257761, -17.7051010, 12.9191923, -33.9619675, 33.0308762
8: -24.2415104, 13.7705793, -20.3778305, 11.6318026, -35.8733139, 34.1484108
9: -16.8565540, 19.7979851, -14.1840363, 16.6682777, -33.5248260, 33.9820137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=89, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=20, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=253, inp2_unstable=250, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912932, upper bound: 27.1909453
time: 5.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912932, upper bound: 27.1909477
time: 4.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -16.4827156, 12.4924955, -16.9858341, 12.8967171, -29.3794327, 29.4783287
1: -13.1600828, 11.1268635, -13.5863924, 11.4882698, -24.6483498, 24.7132568
2: -21.9496689, 7.0267243, -22.5281563, 7.4377155, -29.3873825, 29.5548801
3: -19.5457077, 8.8272667, -20.1509781, 9.1429100, -28.6886101, 28.9782391
4: -19.7918701, 11.9770899, -20.3699417, 12.3765373, -32.1684074, 32.3470306
5: -14.6947384, 12.7177277, -15.1887407, 13.1009407, -27.7956791, 27.9064674
6: -15.6830111, 13.4073477, -16.1784515, 13.8394594, -29.5224686, 29.5858002
7: -17.7191010, 12.8691654, -18.2656956, 13.3297415, -31.0488358, 31.1348610
8: -20.3947258, 11.6285381, -21.0223770, 12.0031166, -32.3978424, 32.6509132
9: -14.1740932, 16.7206497, -14.6338034, 17.1963673, -31.3704586, 31.3544540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=88, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=250, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752181, upper bound: 27.1761430
time: 6.98 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1722802, upper bound: 27.1726510
time: 6.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -16.8072109, 12.7373571, -18.1314869, 13.7519608, -30.5591717, 30.8688431
1: -13.4237556, 11.3432636, -14.5158672, 12.2519855, -25.6757412, 25.8591270
2: -22.3735237, 7.1748500, -24.0264778, 7.9421082, -30.3156319, 31.2013283
3: -19.9325829, 8.9981852, -21.5187569, 9.7410355, -29.6736183, 30.5169411
4: -20.1770668, 12.2089071, -21.7233887, 13.1888561, -33.3659210, 33.9322929
5: -14.9871206, 12.9650249, -16.2159653, 13.9729776, -28.9600983, 29.1809883
6: -15.9927444, 13.6689959, -17.2781448, 14.7655315, -30.7582760, 30.9471397
7: -18.0678749, 13.1230135, -19.4941692, 14.2198210, -32.2876968, 32.6171799
8: -20.7961826, 11.8531303, -22.4435921, 12.7945061, -33.5906830, 34.2967224
9: -14.4528065, 17.0447922, -15.6168671, 18.3410797, -32.7938843, 32.6616516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=24, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=251, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755090, upper bound: 27.1763456
time: 4.03 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1725884, upper bound: 27.1728598
time: 4.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -19.5247192, 14.8095551, -16.6847229, 12.6712198, -32.1959381, 31.4942780
1: -15.6469765, 13.1842670, -13.3410034, 11.2871246, -26.9340992, 26.5252686
2: -25.8448658, 8.5424776, -22.1352348, 7.3016458, -33.1465073, 30.6777115
3: -23.2298965, 10.4788322, -19.7902756, 8.9847498, -32.2146416, 30.2691078
4: -23.4100189, 14.1861391, -20.0131493, 12.1605968, -35.5706062, 34.1992836
5: -17.4934692, 15.0403147, -14.9176588, 12.8726120, -30.3660774, 29.9579735
6: -18.6037083, 15.8899345, -15.8908596, 13.5957794, -32.1994820, 31.7807884
7: -20.9913826, 15.2933502, -17.9432507, 13.0949430, -34.0863228, 33.2365990
8: -24.1858101, 13.7457409, -20.6500072, 11.7940903, -35.9798889, 34.3957481
9: -16.8177071, 19.7527752, -14.3753357, 16.8951721, -33.7128754, 34.1281128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=88, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=253, inp2_unstable=250, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912625, upper bound: 27.1909302
time: 5.55 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912625, upper bound: 27.1909302
time: 8.50 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -19.8576164, 15.0631142, -17.8284798, 13.5247116, -33.3823280, 32.8915939
1: -15.9170017, 13.4089794, -14.2692556, 12.0490637, -27.9660645, 27.6782341
2: -26.2785530, 8.7024059, -23.6308784, 7.8027625, -34.0813141, 32.3332825
3: -23.6231213, 10.6563911, -21.1564140, 9.5811052, -33.2042274, 31.8128052
4: -23.8037701, 14.4273643, -21.3647900, 12.9706621, -36.7744331, 35.7921486
5: -17.7946472, 15.2948818, -15.9428644, 13.7430964, -31.5377312, 31.2377415
6: -18.9276924, 16.1593914, -16.9878063, 14.5202427, -33.4479294, 33.1471977
7: -21.3479652, 15.5552454, -19.1698112, 13.9834003, -35.3313675, 34.7250557
8: -24.6052628, 13.9822197, -22.0679493, 12.5826683, -37.1879311, 36.0501633
9: -17.1053638, 20.0855179, -15.3565264, 18.0384254, -35.1437836, 35.4420433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=23, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=253, inp2_unstable=250, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912740, upper bound: 27.1909400
time: 7.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912740, upper bound: 27.1909400
time: 6.11 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -16.7848148, 12.7463837, -16.7486172, 12.7244434, -29.5092583, 29.4949951
1: -13.4206581, 11.3514471, -13.3982906, 11.3526688, -24.7733250, 24.7497349
2: -22.2612381, 7.3360648, -22.1523628, 7.4361439, -29.6973801, 29.4884186
3: -19.9142666, 9.0360432, -19.8633099, 9.0499077, -28.9641743, 28.8993492
4: -20.1264935, 12.2226000, -20.0551453, 12.2138090, -32.3403015, 32.2777443
5: -15.0083179, 12.9470596, -14.9929352, 12.9211369, -27.9294548, 27.9399910
6: -15.9870701, 13.6770363, -15.9717302, 13.6653500, -29.6524181, 29.6487656
7: -18.0461292, 13.1722403, -18.0138855, 13.1859341, -31.2320633, 31.1861229
8: -20.7694855, 11.8628893, -20.7408447, 11.8494682, -32.6189537, 32.6037292
9: -14.4598036, 16.9970016, -14.4529314, 16.9249992, -31.3848019, 31.4499321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=25, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=251, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927327, upper bound: 27.1927070
time: 9.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927327, upper bound: 27.1927070
time: 5.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -16.4938164, 12.5281315, -20.7298088, 15.7321959, -32.2260132, 33.2579422
1: -13.1829309, 11.1569796, -16.6774101, 14.0822372, -27.2651672, 27.8343887
2: -21.8808537, 7.2055979, -27.1901531, 9.5459099, -31.4267635, 34.3957520
3: -19.5650787, 8.8835812, -24.6946297, 11.2803822, -30.8454609, 33.5782089
4: -19.7808762, 12.0142813, -24.7514801, 15.1584463, -34.9393196, 36.7657623
5: -14.7459335, 12.7260952, -18.6385632, 15.9667454, -30.7126751, 31.3646488
6: -15.7092972, 13.4410744, -19.8284340, 16.9537334, -32.6630325, 33.2695084
7: -17.7343292, 12.9453697, -22.2539272, 16.4043884, -34.1387138, 35.1992874
8: -20.4095192, 11.6607761, -25.7103214, 14.7018328, -35.1113510, 37.3710976
9: -14.2099781, 16.7049637, -17.9404049, 20.8574982, -35.0674744, 34.6453629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=90, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1764142, upper bound: 27.1769159
time: 18.98 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1731947, upper bound: 27.1730404
time: 3.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -17.9134464, 13.5886955, -17.1321144, 13.0159340, -30.9293747, 30.7208099
1: -14.3370094, 12.1032810, -13.7113590, 11.6137381, -25.9507484, 25.8146400
2: -23.7386360, 7.8292828, -22.6482506, 7.6322775, -31.3709106, 30.4775333
3: -21.2624149, 9.6241035, -20.3188705, 9.2575092, -30.5199242, 29.9429741
4: -21.4619350, 13.0206089, -20.5076561, 12.4933710, -33.9553070, 33.5282669
5: -16.0196419, 13.8064632, -15.3415337, 13.2145271, -29.2341690, 29.1479969
6: -17.0694427, 14.5896454, -16.3475609, 13.9794044, -31.0488472, 30.9372025
7: -19.2575645, 14.0489292, -18.4233894, 13.4917746, -32.7493401, 32.4723129
8: -22.1685791, 12.6399736, -21.2236252, 12.1248455, -34.2934265, 33.8635979
9: -15.4282131, 18.1252384, -14.7862730, 17.3059444, -32.7341576, 32.9115105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=27, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=252, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1928051, upper bound: 27.1927796
time: 6.06 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1928051, upper bound: 27.1927796
time: 4.27 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -17.6179695, 13.3672857, -21.1706066, 16.0676804, -33.6856461, 34.5378914
1: -14.0964661, 11.9054756, -17.0333099, 14.3833313, -28.4797974, 28.9387856
2: -23.3529835, 7.6948690, -27.7715530, 9.8016605, -33.1546440, 35.4664230
3: -20.9087181, 9.4687243, -25.2008877, 11.5280018, -32.4367218, 34.6696129
4: -21.1118050, 12.8082027, -25.2487450, 15.4905539, -36.6023560, 38.0569458
5: -15.7535038, 13.5823088, -19.0514755, 16.3041363, -32.0576363, 32.6337852
6: -16.7866821, 14.3503103, -20.2749901, 17.3030357, -34.0897102, 34.6252975
7: -18.9412766, 13.8186646, -22.7232590, 16.7723427, -35.7136192, 36.5419197
8: -21.8027096, 12.4340677, -26.2531052, 15.0228310, -36.8255386, 38.6871719
9: -15.1744480, 17.8298225, -18.3218479, 21.2848129, -36.4592590, 36.1516685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=49, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1765486, upper bound: 27.1771004
time: 9.50 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1734917, upper bound: 27.1734233
time: 4.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -17.0928726, 12.9786110, -18.1451569, 13.7707014, -30.8635750, 31.1237679
1: -13.6705446, 11.5607567, -14.5273552, 12.2838631, -25.9544067, 26.0881119
2: -22.6523476, 7.5002651, -23.9761963, 8.0796852, -30.7320328, 31.4764576
3: -20.2838249, 9.2068090, -21.5348721, 9.7907639, -30.0745888, 30.7416801
4: -20.4843769, 12.4473114, -21.7123833, 13.2202616, -33.7046280, 34.1596909
5: -15.2870407, 13.1795158, -16.2447948, 13.9831371, -29.2701778, 29.4243107
6: -16.2841644, 13.9266672, -17.3167419, 14.7918262, -31.0759907, 31.2434044
7: -18.3731785, 13.4214592, -19.5018654, 14.2763662, -32.6495438, 32.9233246
8: -21.1478519, 12.0830069, -22.4712067, 12.8316269, -33.9794769, 34.5542145
9: -14.7271433, 17.2985573, -15.6506901, 18.3211784, -33.0483208, 32.9492455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=28, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=252, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927096, upper bound: 27.1927048
time: 19.29 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927096, upper bound: 27.1927048
time: 23.25 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -16.7961121, 12.7564793, -21.8703003, 16.6020660, -33.3981781, 34.6267776
1: -13.4287777, 11.3624372, -17.5989552, 14.8497028, -28.2784805, 28.9613914
2: -22.2652435, 7.3662486, -28.6904755, 10.1303310, -32.3955765, 36.0567245
3: -19.9281845, 9.0508976, -26.0523186, 11.9043312, -31.8325157, 35.1032181
4: -20.1327095, 12.2344952, -26.0845413, 15.9975357, -36.1302414, 38.3190384
5: -15.0198059, 12.9544792, -19.6794357, 16.8346786, -31.8544846, 32.6339111
6: -16.0005703, 13.6864929, -20.9479122, 17.8660812, -33.8666496, 34.6344070
7: -18.0552540, 13.1901112, -23.4654331, 17.3205185, -35.3757706, 36.6555405
8: -20.7808342, 11.8769779, -27.1092682, 15.5209465, -36.3017807, 38.9862442
9: -14.4723549, 17.0018311, -18.9201736, 21.9886475, -36.4610023, 35.9220047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=92, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=255, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1761392, upper bound: 27.1768080
time: 4.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1730766, upper bound: 27.1730009
time: 9.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -18.2192554, 13.8199139, -18.5165920, 14.0555582, -32.2748146, 32.3365059
1: -14.5838032, 12.3127098, -14.8323431, 12.5382481, -27.1220512, 27.1450539
2: -24.1251945, 8.0010529, -24.4572105, 8.2749252, -32.4001160, 32.4582634
3: -21.6262245, 9.7959423, -21.9788895, 9.9940033, -31.6202278, 31.7748260
4: -21.8146820, 13.2459440, -22.1519356, 13.4925175, -35.3071976, 35.3978729
5: -16.2973003, 14.0371027, -16.5840836, 14.2682524, -30.5655518, 30.6211853
6: -17.3689480, 14.8372078, -17.6821938, 15.0975380, -32.4664841, 32.5194016
7: -19.5807228, 14.2970600, -19.8986435, 14.5749836, -34.1557045, 34.1957016
8: -22.5482597, 12.8634129, -22.9400272, 13.1008005, -35.6490593, 35.8034363
9: -15.6945677, 18.4227123, -15.9758968, 18.6910133, -34.3855820, 34.3986092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=91, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=30, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=253, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927763, upper bound: 27.1927763
time: 5.18 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927763, upper bound: 27.1927763
time: 6.82 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -17.9213676, 13.5966873, -22.3314304, 16.9991894, -34.9205513, 35.9281158
1: -14.3414116, 12.1132145, -17.9944229, 15.1872368, -29.5286484, 30.1076374
2: -23.7364941, 7.8641362, -29.2784004, 10.4119701, -34.1484604, 37.1425362
3: -21.2700176, 9.6386070, -26.6234856, 12.1715689, -33.4415855, 36.2620926
4: -21.4623222, 13.0314684, -26.6571465, 16.3466854, -37.8090057, 39.6886101
5: -16.0287666, 13.8112116, -20.1347160, 17.2180138, -33.2467804, 33.9459267
6: -17.0835075, 14.5962162, -21.4078197, 18.2673264, -35.3508339, 36.0040359
7: -19.2619362, 14.0647793, -23.9656715, 17.7204056, -36.9823418, 38.0304451
8: -22.1791363, 12.6551447, -27.6938744, 15.8810539, -38.0601883, 40.3490181
9: -15.4386444, 18.1254406, -19.3482780, 22.4597321, -37.8983765, 37.4737053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=93, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=54, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1763100, upper bound: 27.1770084
time: 7.03 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1733822, upper bound: 27.1733822
time: 4.01 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 12.61 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1756639, upper bound: 27.1765012
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1725830, upper bound: 27.1728371
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1758434, upper bound: 27.1766583
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1727857, upper bound: 27.1729945
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1746570, upper bound: 27.1750444
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1715719, upper bound: 27.1711670
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1912932, upper bound: 27.1909453
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1912932, upper bound: 27.1909477
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1752181, upper bound: 27.1761430
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1722802, upper bound: 27.1726510
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1755090, upper bound: 27.1763456
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1725884, upper bound: 27.1728598
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1912625, upper bound: 27.1909302
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1912625, upper bound: 27.1909302
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1912740, upper bound: 27.1909400
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1912740, upper bound: 27.1909400
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1927327, upper bound: 27.1927070
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1927327, upper bound: 27.1927070
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1764142, upper bound: 27.1769159
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1731947, upper bound: 27.1730404
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1928051, upper bound: 27.1927796
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1928051, upper bound: 27.1927796
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1765486, upper bound: 27.1771004
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1734917, upper bound: 27.1734233
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1927096, upper bound: 27.1927048
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1927096, upper bound: 27.1927048
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1761392, upper bound: 27.1768080
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1730766, upper bound: 27.1730009
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1927763, upper bound: 27.1927763
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1927763, upper bound: 27.1927763
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1763100, upper bound: 27.1770084
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.61
Output dim: 2, lower bound: -27.1733822, upper bound: 27.1733822

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -15.9401255, 12.0849628, -15.6298113, 11.8864088, -27.8265305, 27.7147732
1: -12.7191677, 10.7658005, -12.4897137, 10.5909882, -23.3101482, 23.2555084
2: -21.2377720, 6.7723961, -20.7523842, 6.8433084, -28.0810795, 27.5247803
3: -18.8964500, 8.5408030, -18.5304756, 8.4357395, -27.3321896, 27.0712776
4: -19.1518860, 11.5863152, -18.7578888, 11.4063644, -30.5582504, 30.3442020
5: -14.2089872, 12.3064327, -13.9771748, 12.0732813, -26.2822647, 26.2836075
6: -15.1642818, 12.9714594, -14.8856564, 12.7476292, -27.9119110, 27.8571110
7: -17.1360779, 12.4422522, -16.8171005, 12.2798710, -29.4159412, 29.2593498
8: -19.7277718, 11.2486038, -19.3510685, 11.0646057, -30.7923775, 30.5996723
9: -13.7084417, 16.1794186, -13.4754925, 15.8381977, -29.5466385, 29.6549110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=87, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=19, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=246, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1756639, upper bound: 27.1765012
time: 4.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1756639, upper bound: 27.1765012
time: 4.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -16.2616520, 12.3257961, -16.7633343, 12.7316513, -28.9932976, 29.0891304
1: -12.9795742, 10.9794197, -13.4110098, 11.3436203, -24.3231926, 24.3904305
2: -21.6578560, 6.9174185, -22.2385750, 7.3291588, -28.9870148, 29.1559849
3: -19.2800903, 8.7088470, -19.8834877, 9.0235682, -28.3036575, 28.5923347
4: -19.5327034, 11.8155136, -20.0989819, 12.2082129, -31.7409172, 31.9144955
5: -14.4958162, 12.5507994, -14.9917889, 12.9358768, -27.4316902, 27.5425835
6: -15.4705286, 13.2300529, -15.9685974, 13.6642838, -29.1348114, 29.1986504
7: -17.4812393, 12.6928759, -18.0321121, 13.1566267, -30.6378670, 30.7249870
8: -20.1242046, 11.4713440, -20.7552090, 11.8434219, -31.9676266, 32.2265549
9: -13.9843311, 16.4995556, -14.4455919, 16.9747581, -30.9590855, 30.9451408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=89, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=20, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=246, inp2_unstable=250, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1758434, upper bound: 27.1766583
time: 10.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1758434, upper bound: 27.1766583
time: 4.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -18.9619484, 14.3820448, -15.3295527, 11.6617451, -30.6236916, 29.7115974
1: -15.1884060, 12.8066845, -12.2444410, 10.3909225, -25.5793285, 25.0511208
2: -25.1084862, 8.2703114, -20.3610001, 6.7097507, -31.8182373, 28.6313114
3: -22.5589066, 10.1774216, -18.1707001, 8.2786274, -30.8375340, 28.3481216
4: -22.7457809, 13.7782326, -18.4011192, 11.1912985, -33.9370804, 32.1793518
5: -16.9834347, 14.6120481, -13.7066708, 11.8452501, -28.8286743, 28.3187141
6: -18.0613613, 15.4357738, -14.5992002, 12.5039988, -30.5653534, 30.0349731
7: -20.3871632, 14.8479233, -16.4953308, 12.0465832, -32.4337463, 31.3432541
8: -23.4876900, 13.3483925, -18.9794693, 10.8569651, -34.3446541, 32.3278618
9: -16.3332138, 19.1900101, -13.2177391, 15.5371761, -31.8703804, 32.4077454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=86, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=19, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=252, inp2_unstable=248, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1715719, upper bound: 27.1711670
time: 6.07 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1715719, upper bound: 27.1711670
time: 4.32 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -18.3965988, 13.9618187, -16.4579887, 12.5029869, -30.8995781, 30.4198036
1: -14.7351627, 12.4361115, -13.1619911, 11.1399097, -25.8750706, 25.5981026
2: -24.3695984, 8.0422373, -21.8390884, 7.1928687, -31.5624657, 29.8813248
3: -21.8880806, 9.8874016, -19.5180187, 8.8640156, -30.7520943, 29.4054184
4: -22.0773525, 13.3835039, -19.7367096, 11.9895000, -34.0668526, 33.1202126
5: -16.4817219, 14.1808910, -14.7168732, 12.7039700, -29.1856918, 28.8977623
6: -17.5228195, 14.9826221, -15.6771812, 13.4170904, -30.9399033, 30.6598034
7: -19.7800751, 14.4156094, -17.7051010, 12.9191923, -32.6992645, 32.1207123
8: -22.7864819, 12.9652233, -20.3778305, 11.6318026, -34.4182854, 33.3430519
9: -15.8518791, 18.6240158, -14.1840363, 16.6682777, -32.5201569, 32.8080482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=88, inp2_unstable=89, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=20, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=252, inp2_unstable=250, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752709, upper bound: 27.1741871
time: 6.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1715664, upper bound: 27.1712147
time: 5.94 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -19.1809959, 14.5482531, -16.4579887, 12.5029869, -31.6839790, 31.0062408
1: -15.3690996, 12.9543724, -13.1619911, 11.1399097, -26.5090084, 26.1163597
2: -25.4019623, 8.3736982, -21.8390884, 7.1928687, -32.5948296, 30.2127800
3: -22.8216972, 10.2936506, -19.5180187, 8.8640156, -31.6857128, 29.8116684
4: -23.0056992, 13.9364824, -19.7367096, 11.9895000, -34.9951973, 33.6731911
5: -17.1814022, 14.7793951, -14.7168732, 12.7039700, -29.8853683, 29.4962654
6: -18.2712402, 15.6127682, -15.6771812, 13.4170904, -31.6883316, 31.2899494
7: -20.6236115, 15.0210228, -17.7051010, 12.9191923, -33.5428009, 32.7261238
8: -23.7586594, 13.5028687, -20.3778305, 11.6318026, -35.3904610, 33.8806992
9: -16.5217628, 19.4101315, -14.1840363, 16.6682777, -33.1900406, 33.5941658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=89, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=20, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=252, inp2_unstable=250, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752709, upper bound: 27.1743783
time: 5.33 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1715664, upper bound: 27.1714193
time: 22.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -16.2089806, 12.2873306, -16.9858341, 12.8967171, -29.1056976, 29.2731647
1: -12.9369707, 10.9443560, -13.5863924, 11.4882698, -24.4252357, 24.5307484
2: -21.5882263, 6.9008579, -22.5281563, 7.4377155, -29.0259399, 29.4290142
3: -19.2180462, 8.6837187, -20.1509781, 9.1429100, -28.3609543, 28.8346958
4: -19.4676609, 11.7806416, -20.3699417, 12.3765373, -31.8441982, 32.1505814
5: -14.4504128, 12.5096693, -15.1887407, 13.1009407, -27.5513535, 27.6984100
6: -15.4218426, 13.1868601, -16.1784515, 13.8394594, -29.2612991, 29.3653107
7: -17.4240055, 12.6549654, -18.2656956, 13.3297415, -30.7537384, 30.9206619
8: -20.0570927, 11.4375782, -21.0223770, 12.0031166, -32.0602036, 32.4599533
9: -13.9389820, 16.4470501, -14.6338034, 17.1963673, -31.1353493, 31.0808525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=88, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=246, inp2_unstable=250, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752181, upper bound: 27.1761430
time: 7.78 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752181, upper bound: 27.1761430
time: 7.29 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 16.61 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 16.61
Output dim: 2, lower bound: -27.1756639, upper bound: 27.1765012
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 16.61
Output dim: 2, lower bound: -27.1756639, upper bound: 27.1765012
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 16.61
Output dim: 2, lower bound: -27.1758434, upper bound: 27.1766583
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 16.61
Output dim: 2, lower bound: -27.1758434, upper bound: 27.1766583
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 16.61
Output dim: 2, lower bound: -27.1715719, upper bound: 27.1711670
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 16.61
Output dim: 2, lower bound: -27.1715719, upper bound: 27.1711670
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 16.61
Output dim: 2, lower bound: -27.1752709, upper bound: 27.1741871
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 16.61
Output dim: 2, lower bound: -27.1715664, upper bound: 27.1712147
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 16.61
Output dim: 2, lower bound: -27.1752709, upper bound: 27.1743783
IS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 16.61
Output dim: 2, lower bound: -27.1715664, upper bound: 27.1714193
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 16.61
Output dim: 2, lower bound: -27.1752181, upper bound: 27.1761430
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 16.61
Output dim: 2, lower bound: -27.1752181, upper bound: 27.1761430
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1755090, upper bound: 27.1763456
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1912625, upper bound: 27.1909302
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1912625, upper bound: 27.1909302
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1912740, upper bound: 27.1909400
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1912740, upper bound: 27.1909400
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1927327, upper bound: 27.1927070
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1927327, upper bound: 27.1927070
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1764142, upper bound: 27.1769159
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1928051, upper bound: 27.1927796
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1928051, upper bound: 27.1927796
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1765486, upper bound: 27.1771004
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1734917, upper bound: 27.1734233
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1927096, upper bound: 27.1927048
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1927096, upper bound: 27.1927048
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1761392, upper bound: 27.1768080
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1927763, upper bound: 27.1927763
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1927763, upper bound: 27.1927763
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1763100, upper bound: 27.1770084
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 2, lower bound: -27.1733822, upper bound: 27.1733822

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 7.94 + 602.73 = 610.68 seconds
