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
Threshold: 0.04893569308


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0267653, 0.0043655, -0.0267653, 0.0043655, -0.0311308, 0.0311308)
1: (-0.0157691, 0.0103924, -0.0157691, 0.0103924, -0.0261615, 0.0261615)
2: (-0.0008687, 0.0392097, -0.0008687, 0.0392097, -0.0400784, 0.0400784)
3: (-0.0117063, 0.0158329, -0.0117063, 0.0158329, -0.0275392, 0.0275392)
4: (-0.0219076, 0.0200814, -0.0219076, 0.0200814, -0.0419890, 0.0419890)
5: (-0.0133751, 0.0419215, -0.0133751, 0.0419215, -0.0552966, 0.0552966)
6: (-0.0098138, 0.0144340, -0.0098138, 0.0144340, -0.0242479, 0.0242479)
7: (-0.0387665, 0.0121180, -0.0387665, 0.0121180, -0.0508845, 0.0508845)
8: (0.9140067, 1.0133771, 0.9140067, 1.0133771, -0.0993704, 0.0993704)
9: (-0.0052199, 0.0664709, -0.0052199, 0.0664709, -0.0716908, 0.0716908)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.40 + 2.41 = 3.82 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0662909, upper bound: 0.0662909

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0635034, upper bound: 0.0654810
time: 1.28 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0655926, upper bound: 0.0655926
time: 1.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.70 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.70
Output dim: 8, lower bound: -0.0635034, upper bound: 0.0654810
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.70
Output dim: 8, lower bound: -0.0655926, upper bound: 0.0655926

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0050131, 0.0035657, -0.0214771, 0.0037609, -0.0087739, 0.0249880
1: -0.0048818, 0.0062976, -0.0121638, 0.0076125, -0.0124943, 0.0184498
2: 0.0079875, 0.0216462, 0.0047034, 0.0328972, -0.0249098, 0.0169428
3: -0.0093084, 0.0061918, -0.0108924, 0.0119833, -0.0212917, 0.0170842
4: -0.0086276, 0.0080042, -0.0151947, 0.0162660, -0.0248936, 0.0231989
5: -0.0089871, 0.0167602, -0.0120960, 0.0319219, -0.0409090, 0.0288562
6: 0.0005198, 0.0116646, -0.0064086, 0.0116979, -0.0111780, 0.0180732
7: -0.0279559, -0.0036546, -0.0348976, 0.0052650, -0.0332208, 0.0312429
8: 0.9473920, 1.0122538, 0.9315987, 1.0131398, -0.0657478, 0.0806550
9: -0.0045433, 0.0225944, -0.0048707, 0.0508491, -0.0553925, 0.0274650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0635034, upper bound: 0.0635034
time: 2.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0635034, upper bound: 0.0654810
time: 1.26 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0188779, 0.0038043, -0.0234230, 0.0037291, -0.0226070, 0.0272273
1: -0.0108026, 0.0069326, -0.0133116, 0.0085943, -0.0193970, 0.0202442
2: 0.0065275, 0.0300364, 0.0033241, 0.0350142, -0.0284867, 0.0267124
3: -0.0090042, 0.0109885, -0.0105284, 0.0127436, -0.0217478, 0.0215170
4: -0.0120184, 0.0134368, -0.0178958, 0.0167592, -0.0287776, 0.0313325
5: -0.0069228, 0.0294052, -0.0108843, 0.0338639, -0.0407867, 0.0402895
6: -0.0042808, 0.0119670, -0.0074182, 0.0122111, -0.0164920, 0.0193852
7: -0.0333116, 0.0009248, -0.0363095, 0.0080968, -0.0414084, 0.0372343
8: 0.9368077, 1.0159520, 0.9272371, 1.0126936, -0.0758860, 0.0887149
9: -0.0061377, 0.0433947, -0.0048750, 0.0564034, -0.0625411, 0.0482697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0654810, upper bound: 0.0635034
time: 1.75 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0654810, upper bound: 0.0635034
time: 1.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.56 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.56
Output dim: 8, lower bound: -0.0635034, upper bound: 0.0635034
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.56
Output dim: 8, lower bound: -0.0635034, upper bound: 0.0654810
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.56
Output dim: 8, lower bound: -0.0654810, upper bound: 0.0635034
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.56
Output dim: 8, lower bound: -0.0654810, upper bound: 0.0635034

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0050131, 0.0035657, -0.0050131, 0.0035657, -0.0085250, 0.0085250
1: -0.0048818, 0.0062976, -0.0048818, 0.0062976, -0.0111794, 0.0111794
2: 0.0079875, 0.0216462, 0.0079875, 0.0216462, -0.0136587, 0.0136587
3: -0.0093084, 0.0061918, -0.0093084, 0.0061918, -0.0155002, 0.0155002
4: -0.0086276, 0.0080042, -0.0086276, 0.0080042, -0.0166318, 0.0166318
5: -0.0089871, 0.0167602, -0.0089871, 0.0167602, -0.0257473, 0.0257473
6: 0.0005198, 0.0116646, 0.0005198, 0.0116646, -0.0111448, 0.0111448
7: -0.0279559, -0.0036546, -0.0279559, -0.0036546, -0.0243012, 0.0243012
8: 0.9473920, 1.0122538, 0.9473920, 1.0122538, -0.0648617, 0.0648617
9: -0.0045433, 0.0225944, -0.0045433, 0.0225944, -0.0266404, 0.0266404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0622516, upper bound: 0.0619247
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0617230, upper bound: 0.0619182
time: 1.45 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0050131, 0.0035657, -0.0188779, 0.0038043, -0.0088174, 0.0223987
1: -0.0048818, 0.0062976, -0.0108026, 0.0069326, -0.0118145, 0.0171002
2: 0.0079875, 0.0216462, 0.0065275, 0.0300364, -0.0220489, 0.0151187
3: -0.0093084, 0.0061918, -0.0090042, 0.0109885, -0.0202969, 0.0151960
4: -0.0086276, 0.0080042, -0.0120184, 0.0134368, -0.0220644, 0.0200226
5: -0.0089871, 0.0167602, -0.0069228, 0.0294052, -0.0383923, 0.0236830
6: 0.0005198, 0.0116646, -0.0042808, 0.0119670, -0.0114471, 0.0159455
7: -0.0279559, -0.0036546, -0.0333116, 0.0009248, -0.0288807, 0.0296570
8: 0.9473920, 1.0122538, 0.9368077, 1.0159520, -0.0685599, 0.0754461
9: -0.0045433, 0.0225944, -0.0061377, 0.0433947, -0.0479380, 0.0287321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0622516, upper bound: 0.0644332
time: 1.63 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0617230, upper bound: 0.0644025
time: 1.33 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0188779, 0.0038043, -0.0050131, 0.0035657, -0.0223987, 0.0088174
1: -0.0108026, 0.0069326, -0.0048818, 0.0062976, -0.0171002, 0.0118145
2: 0.0065275, 0.0300364, 0.0079875, 0.0216462, -0.0151187, 0.0220489
3: -0.0090042, 0.0109885, -0.0093084, 0.0061918, -0.0151960, 0.0202969
4: -0.0120184, 0.0134368, -0.0086276, 0.0080042, -0.0200226, 0.0220644
5: -0.0069228, 0.0294052, -0.0089871, 0.0167602, -0.0236830, 0.0383923
6: -0.0042808, 0.0119670, 0.0005198, 0.0116646, -0.0159455, 0.0114471
7: -0.0333116, 0.0009248, -0.0279559, -0.0036546, -0.0296570, 0.0288807
8: 0.9368077, 1.0159520, 0.9473920, 1.0122538, -0.0754461, 0.0685599
9: -0.0061377, 0.0433947, -0.0045433, 0.0225944, -0.0287321, 0.0479380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0645215, upper bound: 0.0617295
time: 1.78 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0644025, upper bound: 0.0617230
time: 1.68 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0188779, 0.0038043, -0.0188779, 0.0038043, -0.0226822, 0.0226822
1: -0.0108026, 0.0069326, -0.0108026, 0.0069326, -0.0177352, 0.0177352
2: 0.0065275, 0.0300364, 0.0065275, 0.0300364, -0.0235089, 0.0235089
3: -0.0090042, 0.0109885, -0.0090042, 0.0109885, -0.0199927, 0.0199927
4: -0.0120184, 0.0134368, -0.0120184, 0.0134368, -0.0254552, 0.0254552
5: -0.0069228, 0.0294052, -0.0069228, 0.0294052, -0.0363280, 0.0363280
6: -0.0042808, 0.0119670, -0.0042808, 0.0119670, -0.0162478, 0.0162478
7: -0.0333116, 0.0009248, -0.0333116, 0.0009248, -0.0342364, 0.0342364
8: 0.9368077, 1.0159520, 0.9368077, 1.0159520, -0.0791443, 0.0791443
9: -0.0061377, 0.0433947, -0.0061377, 0.0433947, -0.0495324, 0.0495324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0645215, upper bound: 0.0633047
time: 1.37 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0644025, upper bound: 0.0633019
time: 1.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.33 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.33
Output dim: 8, lower bound: -0.0622516, upper bound: 0.0619247
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.33
Output dim: 8, lower bound: -0.0617230, upper bound: 0.0619182
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.33
Output dim: 8, lower bound: -0.0622516, upper bound: 0.0644332
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.33
Output dim: 8, lower bound: -0.0617230, upper bound: 0.0644025
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.33
Output dim: 8, lower bound: -0.0645215, upper bound: 0.0617295
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.33
Output dim: 8, lower bound: -0.0644025, upper bound: 0.0617230
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.33
Output dim: 8, lower bound: -0.0645215, upper bound: 0.0633047
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.33
Output dim: 8, lower bound: -0.0644025, upper bound: 0.0633019

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0016178, 0.0035258, -0.0044269, 0.0035592, -0.0051299, 0.0078950
1: -0.0030040, 0.0062450, -0.0045586, 0.0062889, -0.0092928, 0.0108036
2: 0.0080633, 0.0214249, 0.0080004, 0.0216093, -0.0135460, 0.0134245
3: -0.0091174, 0.0047613, -0.0092770, 0.0059333, -0.0150507, 0.0140383
4: -0.0085859, 0.0063107, -0.0086204, 0.0077066, -0.0162924, 0.0149311
5: -0.0084668, 0.0134315, -0.0088983, 0.0162035, -0.0246703, 0.0223298
6: 0.0009011, 0.0116549, 0.0005958, 0.0116630, -0.0107619, 0.0110591
7: -0.0262134, -0.0037877, -0.0276645, -0.0036770, -0.0225364, 0.0237713
8: 0.9496382, 1.0119457, 0.9477855, 1.0122032, -0.0625650, 0.0641602
9: -0.0044963, 0.0162654, -0.0045356, 0.0214721, -0.0253739, 0.0208011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0581886, upper bound: 0.0592985
time: 1.41 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0580403, upper bound: 0.0576028
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0009181, 0.0037769, -0.0027340, 0.0035489, -0.0044466, 0.0065108
1: -0.0026040, 0.0064955, -0.0036251, 0.0062733, -0.0088772, 0.0101206
2: 0.0079067, 0.0214903, 0.0080311, 0.0215067, -0.0136000, 0.0134592
3: -0.0104294, 0.0046021, -0.0092252, 0.0051868, -0.0156162, 0.0138273
4: -0.0086978, 0.0074890, -0.0086042, 0.0069155, -0.0156133, 0.0160933
5: -0.0114970, 0.0125223, -0.0087271, 0.0145955, -0.0260926, 0.0212493
6: 0.0001457, 0.0117006, 0.0007556, 0.0116603, -0.0115146, 0.0109450
7: -0.0255839, -0.0031230, -0.0268227, -0.0037163, -0.0218676, 0.0236997
8: 0.9504898, 1.0136420, 0.9489219, 1.0121194, -0.0616296, 0.0647200
9: -0.0047178, 0.0161841, -0.0045224, 0.0182678, -0.0227204, 0.0207065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0577072, upper bound: 0.0592476
time: 1.41 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0575879, upper bound: 0.0575879
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0016178, 0.0035258, -0.0180576, 0.0037979, -0.0054157, 0.0215346
1: -0.0030040, 0.0062450, -0.0104638, 0.0068745, -0.0098784, 0.0167008
2: 0.0080633, 0.0214249, 0.0068706, 0.0293190, -0.0212557, 0.0145544
3: -0.0091174, 0.0047613, -0.0089545, 0.0106698, -0.0197872, 0.0137158
4: -0.0085859, 0.0063107, -0.0113290, 0.0130418, -0.0216277, 0.0176397
5: -0.0084668, 0.0134315, -0.0068354, 0.0285932, -0.0370600, 0.0202669
6: 0.0009011, 0.0116549, -0.0037875, 0.0119634, -0.0110623, 0.0154424
7: -0.0262134, -0.0037877, -0.0329934, -0.0002083, -0.0260051, 0.0292057
8: 0.9496382, 1.0119457, 0.9381357, 1.0159047, -0.0662664, 0.0738100
9: -0.0044963, 0.0162654, -0.0061206, 0.0419149, -0.0464113, 0.0223860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0580380, upper bound: 0.0564720
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0618968, upper bound: 0.0642028
time: 1.94 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0009181, 0.0037769, -0.0159342, 0.0037885, -0.0046302, 0.0197111
1: -0.0026040, 0.0064955, -0.0095861, 0.0068104, -0.0094144, 0.0160816
2: 0.0079067, 0.0214903, 0.0068932, 0.0278689, -0.0195021, 0.0145971
3: -0.0104294, 0.0046021, -0.0088568, 0.0099512, -0.0203806, 0.0134590
4: -0.0086978, 0.0074890, -0.0096430, 0.0120486, -0.0207464, 0.0171320
5: -0.0114970, 0.0125223, -0.0066774, 0.0266583, -0.0381553, 0.0191996
6: 0.0001457, 0.0117006, -0.0025538, 0.0119598, -0.0118141, 0.0142544
7: -0.0255839, -0.0031230, -0.0321976, -0.0018731, -0.0237109, 0.0290746
8: 0.9504898, 1.0136420, 0.9398698, 1.0158275, -0.0653377, 0.0737721
9: -0.0047178, 0.0161841, -0.0060828, 0.0387313, -0.0425435, 0.0222669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0577154, upper bound: 0.0564304
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0613666, upper bound: 0.0641714
time: 1.38 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0139309, 0.0037671, -0.0044269, 0.0035592, -0.0174441, 0.0080800
1: -0.0087593, 0.0067750, -0.0045586, 0.0062889, -0.0150481, 0.0113336
2: 0.0069243, 0.0265476, 0.0080004, 0.0216093, -0.0146851, 0.0185472
3: -0.0087021, 0.0092897, -0.0092770, 0.0059333, -0.0146355, 0.0185667
4: -0.0094402, 0.0110622, -0.0086204, 0.0077066, -0.0171468, 0.0196826
5: -0.0063882, 0.0248609, -0.0088983, 0.0162035, -0.0225917, 0.0337592
6: -0.0014467, 0.0119548, 0.0005958, 0.0116630, -0.0131098, 0.0113590
7: -0.0314521, -0.0023808, -0.0276645, -0.0036770, -0.0277751, 0.0252836
8: 0.9412497, 1.0156634, 0.9477855, 1.0122032, -0.0709535, 0.0678779
9: -0.0060344, 0.0357789, -0.0045356, 0.0214721, -0.0275065, 0.0395785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0589819, upper bound: 0.0589492
time: 1.23 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0587487, upper bound: 0.0573630
time: 1.51 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0107534, 0.0040011, -0.0027340, 0.0035489, -0.0142691, 0.0066916
1: -0.0074289, 0.0070334, -0.0036251, 0.0062733, -0.0137022, 0.0106585
2: 0.0067533, 0.0246624, 0.0080311, 0.0215067, -0.0147534, 0.0166314
3: -0.0098968, 0.0082309, -0.0092252, 0.0051868, -0.0150836, 0.0174560
4: -0.0095354, 0.0107533, -0.0086042, 0.0069155, -0.0164509, 0.0193576
5: -0.0092319, 0.0219684, -0.0087271, 0.0145955, -0.0238275, 0.0306955
6: -0.0010624, 0.0120039, 0.0007556, 0.0116603, -0.0127228, 0.0112483
7: -0.0302525, -0.0018420, -0.0268227, -0.0037163, -0.0265362, 0.0249808
8: 0.9434702, 1.0173756, 0.9489219, 1.0121194, -0.0686492, 0.0684537
9: -0.0062481, 0.0317460, -0.0045224, 0.0182678, -0.0245159, 0.0356445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0584298, upper bound: 0.0589300
time: 1.26 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0582688, upper bound: 0.0573469
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0139309, 0.0037671, -0.0180576, 0.0037979, -0.0177289, 0.0215799
1: -0.0087593, 0.0067750, -0.0104638, 0.0068745, -0.0156337, 0.0172388
2: 0.0069243, 0.0265476, 0.0068706, 0.0293190, -0.0223948, 0.0196771
3: -0.0087021, 0.0092897, -0.0089545, 0.0106698, -0.0193719, 0.0182442
4: -0.0094402, 0.0110622, -0.0113290, 0.0130418, -0.0224820, 0.0223911
5: -0.0063882, 0.0248609, -0.0068354, 0.0285932, -0.0349813, 0.0316963
6: -0.0014467, 0.0119548, -0.0037875, 0.0119634, -0.0134101, 0.0157423
7: -0.0314521, -0.0023808, -0.0329934, -0.0002083, -0.0312438, 0.0306126
8: 0.9412497, 1.0156634, 0.9381357, 1.0159047, -0.0742723, 0.0775276
9: -0.0060344, 0.0357789, -0.0061206, 0.0419149, -0.0479493, 0.0418994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0610211, upper bound: 0.0559281
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0644408, upper bound: 0.0629073
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0107534, 0.0040011, -0.0159342, 0.0037885, -0.0143116, 0.0197550
1: -0.0074289, 0.0070334, -0.0095861, 0.0068104, -0.0142393, 0.0166195
2: 0.0067533, 0.0246624, 0.0068932, 0.0278689, -0.0211155, 0.0177692
3: -0.0098968, 0.0082309, -0.0088568, 0.0099512, -0.0198480, 0.0170877
4: -0.0095354, 0.0107533, -0.0096430, 0.0120486, -0.0215840, 0.0203963
5: -0.0092319, 0.0219684, -0.0066774, 0.0266583, -0.0358902, 0.0286458
6: -0.0010624, 0.0120039, -0.0025538, 0.0119598, -0.0130223, 0.0145576
7: -0.0302525, -0.0018420, -0.0321976, -0.0018731, -0.0283795, 0.0303556
8: 0.9434702, 1.0173756, 0.9398698, 1.0158275, -0.0723574, 0.0775058
9: -0.0062481, 0.0317460, -0.0060828, 0.0387313, -0.0439990, 0.0378288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0606405, upper bound: 0.0559020
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0643079, upper bound: 0.0629060
time: 1.27 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.64 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 8, lower bound: -0.0581886, upper bound: 0.0592985
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 8, lower bound: -0.0580403, upper bound: 0.0576028
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 8, lower bound: -0.0577072, upper bound: 0.0592476
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 8, lower bound: -0.0575879, upper bound: 0.0575879
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 8, lower bound: -0.0580380, upper bound: 0.0564720
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 8, lower bound: -0.0618968, upper bound: 0.0642028
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 8, lower bound: -0.0577154, upper bound: 0.0564304
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 8, lower bound: -0.0613666, upper bound: 0.0641714
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 8, lower bound: -0.0589819, upper bound: 0.0589492
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 8, lower bound: -0.0587487, upper bound: 0.0573630
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 8, lower bound: -0.0584298, upper bound: 0.0589300
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 8, lower bound: -0.0582688, upper bound: 0.0573469
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 8, lower bound: -0.0610211, upper bound: 0.0559281
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 8, lower bound: -0.0644408, upper bound: 0.0629073
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 8, lower bound: -0.0606405, upper bound: 0.0559020
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 8, lower bound: -0.0643079, upper bound: 0.0629060

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0016178, 0.0035258, -0.0042159, 0.0033481, -0.0048818, 0.0076832
1: -0.0030040, 0.0062450, -0.0044578, 0.0060629, -0.0090669, 0.0107028
2: 0.0080633, 0.0214249, 0.0081476, 0.0214286, -0.0133652, 0.0132773
3: -0.0091174, 0.0047613, -0.0081981, 0.0058493, -0.0149667, 0.0129594
4: -0.0085859, 0.0063107, -0.0085208, 0.0065746, -0.0151605, 0.0148316
5: -0.0084668, 0.0134315, -0.0064767, 0.0160298, -0.0244967, 0.0199083
6: 0.0009011, 0.0116549, 0.0012915, 0.0116237, -0.0107226, 0.0103634
7: -0.0262134, -0.0037877, -0.0275736, -0.0042653, -0.0219480, 0.0236402
8: 0.9496382, 1.0119457, 0.9479083, 1.0107148, -0.0610765, 0.0640374
9: -0.0044963, 0.0162654, -0.0043452, 0.0206394, -0.0244624, 0.0206106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0580403, upper bound: 0.0576028
time: 2.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0580403, upper bound: 0.0576028
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0014825, 0.0033467, -0.0236739, 0.0029816, -0.0043615, 0.0269418
1: -0.0029382, 0.0060434, -0.0152183, 0.0056360, -0.0085742, 0.0210847
2: 0.0082129, 0.0212869, -0.0009983, 0.0223155, -0.0141026, 0.0222852
3: -0.0083001, 0.0047484, -0.0066709, 0.0144476, -0.0227477, 0.0114193
4: -0.0084831, 0.0053830, -0.0088817, 0.0172179, -0.0252425, 0.0142647
5: -0.0066717, 0.0132622, -0.0036755, 0.0345643, -0.0412360, 0.0169377
6: 0.0014437, 0.0116130, -0.0051574, 0.0115082, -0.0100645, 0.0166721
7: -0.0261247, -0.0043062, -0.0372760, -0.0001404, -0.0259844, 0.0326435
8: 0.9496948, 1.0106032, 0.9348087, 1.0076886, -0.0579939, 0.0757945
9: -0.0042931, 0.0157040, -0.0037858, 0.0577528, -0.0610838, 0.0194898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0502235, upper bound: 0.0529826
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0575841, upper bound: 0.0571431
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0009181, 0.0037769, -0.0025234, 0.0033381, -0.0041988, 0.0063002
1: -0.0026040, 0.0064955, -0.0035247, 0.0060483, -0.0086522, 0.0100203
2: 0.0079067, 0.0214903, 0.0081748, 0.0213240, -0.0134174, 0.0133155
3: -0.0104294, 0.0046021, -0.0081479, 0.0051031, -0.0155325, 0.0127501
4: -0.0086978, 0.0074890, -0.0085062, 0.0057377, -0.0144355, 0.0159953
5: -0.0114970, 0.0125223, -0.0063107, 0.0144226, -0.0259196, 0.0188330
6: 0.0001457, 0.0117006, 0.0014416, 0.0116210, -0.0114753, 0.0102590
7: -0.0255839, -0.0031230, -0.0267322, -0.0043014, -0.0212825, 0.0236092
8: 0.9504898, 1.0136420, 0.9490441, 1.0106333, -0.0601435, 0.0645978
9: -0.0047178, 0.0161841, -0.0043319, 0.0174040, -0.0218036, 0.0205161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0575879, upper bound: 0.0575879
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0575879, upper bound: 0.0575879
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0008784, 0.0035970, -0.0219795, 0.0029728, -0.0037740, 0.0255630
1: -0.0025957, 0.0062973, -0.0142844, 0.0056232, -0.0082189, 0.0205195
2: 0.0080516, 0.0213476, 0.0006000, 0.0222078, -0.0141563, 0.0207476
3: -0.0096072, 0.0045883, -0.0066298, 0.0137007, -0.0233079, 0.0112180
4: -0.0085973, 0.0065981, -0.0082834, 0.0161070, -0.0242516, 0.0148814
5: -0.0097023, 0.0125130, -0.0035046, 0.0329559, -0.0426582, 0.0160177
6: 0.0006813, 0.0116596, -0.0042426, 0.0115057, -0.0108244, 0.0157319
7: -0.0255639, -0.0036345, -0.0364340, -0.0019524, -0.0236115, 0.0327995
8: 0.9505473, 1.0123152, 0.9359456, 1.0076159, -0.0570686, 0.0763695
9: -0.0045189, 0.0157365, -0.0037734, 0.0544387, -0.0579835, 0.0195099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0552109, upper bound: 0.0552867
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0574685, upper bound: 0.0574685
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0008313, 0.0035055, -0.0036629, 0.0039282, -0.0046920, 0.0071226
1: -0.0025711, 0.0062139, -0.0041302, 0.0069449, -0.0095161, 0.0103441
2: 0.0080903, 0.0212125, 0.0068443, 0.0216100, -0.0135197, 0.0143682
3: -0.0090416, 0.0045555, -0.0092830, 0.0055923, -0.0146338, 0.0138386
4: -0.0085704, 0.0058755, -0.0094270, 0.0070838, -0.0156542, 0.0153025
5: -0.0083073, 0.0124854, -0.0077622, 0.0154656, -0.0237728, 0.0202476
6: 0.0010054, 0.0116486, 0.0000494, 0.0119952, -0.0109898, 0.0115992
7: -0.0255038, -0.0038575, -0.0272782, -0.0021091, -0.0233947, 0.0234207
8: 0.9507193, 1.0117757, 0.9483070, 1.0168808, -0.0661615, 0.0634687
9: -0.0044657, 0.0153334, -0.0061448, 0.0200916, -0.0241251, 0.0214782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0534654, upper bound: 0.0486609
time: 1.29 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0522276, upper bound: 0.0485085
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0016178, 0.0035258, -0.0156272, 0.0037840, -0.0052968, 0.0190715
1: -0.0030040, 0.0062450, -0.0094602, 0.0068012, -0.0098052, 0.0155579
2: 0.0080633, 0.0214249, 0.0068968, 0.0276613, -0.0190045, 0.0145281
3: -0.0091174, 0.0047613, -0.0088205, 0.0098502, -0.0189676, 0.0135818
4: -0.0085859, 0.0063107, -0.0094760, 0.0118670, -0.0204529, 0.0157867
5: -0.0084668, 0.0134315, -0.0065979, 0.0263846, -0.0348514, 0.0200294
6: 0.0009011, 0.0116549, -0.0023648, 0.0119596, -0.0110584, 0.0140198
7: -0.0262134, -0.0037877, -0.0320840, -0.0019851, -0.0242282, 0.0278093
8: 0.9496382, 1.0119457, 0.9400799, 1.0157975, -0.0661592, 0.0718658
9: -0.0044963, 0.0162654, -0.0060772, 0.0382619, -0.0415435, 0.0223426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0588483, upper bound: 0.0579148
time: 3.05 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0573626, upper bound: 0.0577606
time: 1.37 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0008037, 0.0037522, -0.0020343, 0.0039188, -0.0046674, 0.0057865
1: -0.0024420, 0.0064654, -0.0032318, 0.0069311, -0.0093731, 0.0096972
2: 0.0079284, 0.0212647, 0.0068688, 0.0215135, -0.0135851, 0.0143959
3: -0.0103306, 0.0044101, -0.0092353, 0.0048739, -0.0152045, 0.0136454
4: -0.0086827, 0.0073054, -0.0094138, 0.0062575, -0.0149402, 0.0167191
5: -0.0112697, 0.0123403, -0.0076034, 0.0139181, -0.0251878, 0.0199437
6: 0.0002303, 0.0116945, 0.0001760, 0.0119927, -0.0117623, 0.0115185
7: -0.0251887, -0.0031975, -0.0264681, -0.0021437, -0.0230451, 0.0232706
8: 0.9516220, 1.0134534, 0.9494008, 1.0168037, -0.0651817, 0.0640526
9: -0.0046879, 0.0158815, -0.0061323, 0.0170046, -0.0216926, 0.0220138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0531497, upper bound: 0.0486342
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0519943, upper bound: 0.0484720
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0009181, 0.0037769, -0.0135553, 0.0037748, -0.0046146, 0.0173139
1: -0.0026040, 0.0064955, -0.0086040, 0.0067832, -0.0093872, 0.0150821
2: 0.0079067, 0.0214903, 0.0069190, 0.0263077, -0.0178598, 0.0145713
3: -0.0104294, 0.0046021, -0.0087250, 0.0091655, -0.0195949, 0.0133271
4: -0.0086978, 0.0074890, -0.0094407, 0.0109007, -0.0195985, 0.0169297
5: -0.0114970, 0.0125223, -0.0064435, 0.0245232, -0.0360202, 0.0189658
6: 0.0001457, 0.0117006, -0.0012795, 0.0119568, -0.0118110, 0.0129802
7: -0.0255839, -0.0031230, -0.0313120, -0.0023997, -0.0231842, 0.0280752
8: 0.9504898, 1.0136420, 0.9415089, 1.0157213, -0.0652315, 0.0721331
9: -0.0047178, 0.0161841, -0.0060404, 0.0352478, -0.0387544, 0.0222245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0585239, upper bound: 0.0578922
time: 1.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0568946, upper bound: 0.0577309
time: 1.49 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0139309, 0.0037671, -0.0042159, 0.0033481, -0.0171960, 0.0078682
1: -0.0087593, 0.0067750, -0.0044578, 0.0060629, -0.0148222, 0.0112328
2: 0.0069243, 0.0265476, 0.0081476, 0.0214286, -0.0145043, 0.0184000
3: -0.0087021, 0.0092897, -0.0081981, 0.0058493, -0.0145515, 0.0174878
4: -0.0094402, 0.0110622, -0.0085208, 0.0065746, -0.0160148, 0.0195830
5: -0.0063882, 0.0248609, -0.0064767, 0.0160298, -0.0224180, 0.0313376
6: -0.0014467, 0.0119548, 0.0012915, 0.0116237, -0.0130705, 0.0106633
7: -0.0314521, -0.0023808, -0.0275736, -0.0042653, -0.0271867, 0.0251927
8: 0.9412497, 1.0156634, 0.9479083, 1.0107148, -0.0694650, 0.0677551
9: -0.0060344, 0.0357789, -0.0043452, 0.0206394, -0.0266738, 0.0393370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0506005, upper bound: 0.0536511
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0584560, upper bound: 0.0585431
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0136974, 0.0035998, -0.0236739, 0.0029816, -0.0165763, 0.0271389
1: -0.0086730, 0.0065778, -0.0152183, 0.0056360, -0.0143090, 0.0217961
2: 0.0070829, 0.0262939, -0.0009983, 0.0223155, -0.0152326, 0.0272922
3: -0.0078797, 0.0092177, -0.0066709, 0.0144476, -0.0223273, 0.0158886
4: -0.0093178, 0.0103444, -0.0088817, 0.0172179, -0.0265357, 0.0192261
5: -0.0045462, 0.0246732, -0.0036755, 0.0345643, -0.0391105, 0.0283487
6: -0.0009506, 0.0119144, -0.0051574, 0.0115082, -0.0124589, 0.0170718
7: -0.0313743, -0.0029668, -0.0372760, -0.0001404, -0.0312339, 0.0343092
8: 0.9413937, 1.0143601, 0.9348087, 1.0076886, -0.0662949, 0.0795513
9: -0.0058216, 0.0351164, -0.0037858, 0.0577528, -0.0635745, 0.0389022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0501724, upper bound: 0.0524839
time: 1.29 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0582204, upper bound: 0.0569096
time: 1.33 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0107534, 0.0040011, -0.0025234, 0.0033381, -0.0140213, 0.0064806
1: -0.0074289, 0.0070334, -0.0035247, 0.0060483, -0.0134771, 0.0105581
2: 0.0067533, 0.0246624, 0.0081748, 0.0213240, -0.0145707, 0.0164877
3: -0.0098968, 0.0082309, -0.0081479, 0.0051031, -0.0149999, 0.0163788
4: -0.0095354, 0.0107533, -0.0085062, 0.0057377, -0.0152731, 0.0192595
5: -0.0092319, 0.0219684, -0.0063107, 0.0144226, -0.0236545, 0.0282791
6: -0.0010624, 0.0120039, 0.0014416, 0.0116210, -0.0126834, 0.0105623
7: -0.0302525, -0.0018420, -0.0267322, -0.0043014, -0.0259511, 0.0248902
8: 0.9434702, 1.0173756, 0.9490441, 1.0106333, -0.0671632, 0.0683315
9: -0.0062481, 0.0317460, -0.0043319, 0.0174040, -0.0236521, 0.0354040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0582688, upper bound: 0.0573469
time: 1.10 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0582688, upper bound: 0.0573469
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0104992, 0.0038275, -0.0219795, 0.0029728, -0.0133820, 0.0257359
1: -0.0073344, 0.0068313, -0.0142844, 0.0056232, -0.0129576, 0.0211158
2: 0.0069102, 0.0243800, 0.0006000, 0.0222078, -0.0152976, 0.0237800
3: -0.0090251, 0.0081522, -0.0066298, 0.0137007, -0.0227258, 0.0147820
4: -0.0094190, 0.0099198, -0.0082834, 0.0161070, -0.0255260, 0.0182032
5: -0.0072957, 0.0217630, -0.0035046, 0.0329559, -0.0402516, 0.0252676
6: -0.0004580, 0.0119631, -0.0042426, 0.0115057, -0.0119637, 0.0162058
7: -0.0301673, -0.0023906, -0.0364340, -0.0019524, -0.0282150, 0.0340434
8: 0.9436280, 1.0160495, 0.9359456, 1.0076159, -0.0639879, 0.0801039
9: -0.0060395, 0.0310124, -0.0037734, 0.0544387, -0.0604782, 0.0347858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0484720, upper bound: 0.0519943
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0577309, upper bound: 0.0568946
time: 1.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0092562, 0.0037476, -0.0036629, 0.0039282, -0.0129586, 0.0071603
1: -0.0068274, 0.0067434, -0.0041302, 0.0069449, -0.0132620, 0.0108736
2: 0.0069741, 0.0234930, 0.0068443, 0.0216100, -0.0146359, 0.0163955
3: -0.0084937, 0.0077449, -0.0092830, 0.0055923, -0.0140859, 0.0170279
4: -0.0093611, 0.0089047, -0.0094270, 0.0070838, -0.0164449, 0.0183318
5: -0.0060332, 0.0206607, -0.0077622, 0.0154656, -0.0214987, 0.0284229
6: 0.0000961, 0.0119486, 0.0000494, 0.0119952, -0.0118991, 0.0118992
7: -0.0297101, -0.0026316, -0.0272782, -0.0021091, -0.0261068, 0.0246466
8: 0.9444741, 1.0154976, 0.9483070, 1.0168808, -0.0721747, 0.0671905
9: -0.0059518, 0.0289742, -0.0061448, 0.0200916, -0.0252284, 0.0329293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0583594, upper bound: 0.0534025
time: 1.36 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0608796, upper bound: 0.0557948
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0139309, 0.0037671, -0.0156272, 0.0037840, -0.0174722, 0.0191204
1: -0.0087593, 0.0067750, -0.0094602, 0.0068012, -0.0155605, 0.0162352
2: 0.0069243, 0.0265476, 0.0068968, 0.0276613, -0.0207371, 0.0196508
3: -0.0087021, 0.0092897, -0.0088205, 0.0098502, -0.0185523, 0.0181102
4: -0.0094402, 0.0110622, -0.0094760, 0.0118670, -0.0213072, 0.0205381
5: -0.0063882, 0.0248609, -0.0065979, 0.0263846, -0.0327728, 0.0314588
6: -0.0014467, 0.0119548, -0.0023648, 0.0119596, -0.0134063, 0.0143197
7: -0.0314521, -0.0023808, -0.0320840, -0.0019851, -0.0294669, 0.0297032
8: 0.9412497, 1.0156634, 0.9400799, 1.0157975, -0.0741114, 0.0737116
9: -0.0060344, 0.0357789, -0.0060772, 0.0382619, -0.0434589, 0.0417063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0574606, upper bound: 0.0597248
time: 1.24 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0574606, upper bound: 0.0629073
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0060066, 0.0039769, -0.0020343, 0.0039188, -0.0097113, 0.0058268
1: -0.0054175, 0.0069961, -0.0032318, 0.0069311, -0.0121648, 0.0102279
2: 0.0068119, 0.0218137, 0.0068688, 0.0215135, -0.0147015, 0.0148174
3: -0.0096556, 0.0066228, -0.0092353, 0.0048739, -0.0145295, 0.0158581
4: -0.0094435, 0.0086299, -0.0094138, 0.0062575, -0.0157010, 0.0180437
5: -0.0087806, 0.0176829, -0.0076034, 0.0139181, -0.0226987, 0.0252863
6: -0.0003195, 0.0119976, 0.0001760, 0.0119927, -0.0123122, 0.0118215
7: -0.0284389, -0.0019736, -0.0264681, -0.0021437, -0.0246359, 0.0244317
8: 0.9467399, 1.0171821, 0.9494008, 1.0168037, -0.0700639, 0.0677813
9: -0.0061560, 0.0247331, -0.0061323, 0.0170046, -0.0222771, 0.0289007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0543782, upper bound: 0.0484752
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0526644, upper bound: 0.0482632
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0107534, 0.0040011, -0.0135553, 0.0037748, -0.0142960, 0.0173478
1: -0.0074289, 0.0070334, -0.0086040, 0.0067832, -0.0142121, 0.0156374
2: 0.0067533, 0.0246624, 0.0069190, 0.0263077, -0.0195544, 0.0177434
3: -0.0098968, 0.0082309, -0.0087250, 0.0091655, -0.0190623, 0.0169558
4: -0.0095354, 0.0107533, -0.0094407, 0.0109007, -0.0204362, 0.0201940
5: -0.0092319, 0.0219684, -0.0064435, 0.0245232, -0.0337551, 0.0284119
6: -0.0010624, 0.0120039, -0.0012795, 0.0119568, -0.0130192, 0.0132834
7: -0.0302525, -0.0018420, -0.0313120, -0.0023997, -0.0278528, 0.0294701
8: 0.9434702, 1.0173756, 0.9415089, 1.0157213, -0.0722511, 0.0752280
9: -0.0062481, 0.0317460, -0.0060404, 0.0352478, -0.0401835, 0.0374014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0603730, upper bound: 0.0575025
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0577299, upper bound: 0.0572011
time: 1.07 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.53 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0580403, upper bound: 0.0576028
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0580403, upper bound: 0.0576028
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0502235, upper bound: 0.0529826
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0575841, upper bound: 0.0571431
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0575879, upper bound: 0.0575879
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0575879, upper bound: 0.0575879
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0552109, upper bound: 0.0552867
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0574685, upper bound: 0.0574685
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0534654, upper bound: 0.0486609
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0522276, upper bound: 0.0485085
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0588483, upper bound: 0.0579148
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0573626, upper bound: 0.0577606
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0531497, upper bound: 0.0486342
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0519943, upper bound: 0.0484720
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0585239, upper bound: 0.0578922
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0568946, upper bound: 0.0577309
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0506005, upper bound: 0.0536511
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0584560, upper bound: 0.0585431
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0501724, upper bound: 0.0524839
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0582204, upper bound: 0.0569096
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0582688, upper bound: 0.0573469
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0582688, upper bound: 0.0573469
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0484720, upper bound: 0.0519943
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0577309, upper bound: 0.0568946
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0583594, upper bound: 0.0534025
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0608796, upper bound: 0.0557948
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0574606, upper bound: 0.0597248
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0574606, upper bound: 0.0629073
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0543782, upper bound: 0.0484752
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0526644, upper bound: 0.0482632
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0603730, upper bound: 0.0575025
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0577299, upper bound: 0.0572011

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0014686, 0.0033161, -0.0042159, 0.0033481, -0.0047327, 0.0074361
1: -0.0029347, 0.0060212, -0.0044578, 0.0060629, -0.0089976, 0.0104790
2: 0.0082038, 0.0212416, 0.0081476, 0.0214286, -0.0132248, 0.0130940
3: -0.0080429, 0.0047469, -0.0081981, 0.0058493, -0.0138922, 0.0129450
4: -0.0084896, 0.0050842, -0.0085208, 0.0065746, -0.0150642, 0.0136050
5: -0.0060643, 0.0132532, -0.0064767, 0.0160298, -0.0220941, 0.0197300
6: 0.0015789, 0.0116156, 0.0012915, 0.0116237, -0.0100448, 0.0103241
7: -0.0261200, -0.0043699, -0.0275736, -0.0042653, -0.0218547, 0.0228656
8: 0.9496977, 1.0104638, 0.9479083, 1.0107148, -0.0610171, 0.0625556
9: -0.0043059, 0.0155469, -0.0043452, 0.0206394, -0.0242223, 0.0198839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0536679, upper bound: 0.0513409
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0577316, upper bound: 0.0589097
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0139164, 0.0029510, -0.0042159, 0.0033481, -0.0171727, 0.0070343
1: -0.0101451, 0.0055963, -0.0044578, 0.0060629, -0.0161267, 0.0100541
2: 0.0085520, 0.0221228, 0.0081476, 0.0214286, -0.0128766, 0.0139752
3: -0.0065285, 0.0058755, -0.0081981, 0.0058493, -0.0123778, 0.0140736
4: -0.0082081, 0.0152729, -0.0085208, 0.0065746, -0.0147827, 0.0236713
5: -0.0023325, 0.0318078, -0.0064767, 0.0160298, -0.0183623, 0.0382846
6: -0.0004441, 0.0115007, 0.0012915, 0.0116237, -0.0120679, 0.0102092
7: -0.0358330, -0.0054864, -0.0275736, -0.0042653, -0.0315677, 0.0220871
8: 0.9435127, 1.0074497, 0.9479083, 1.0107148, -0.0672021, 0.0595415
9: -0.0037491, 0.0317911, -0.0043452, 0.0206394, -0.0243885, 0.0352032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0536679, upper bound: 0.0513409
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0577316, upper bound: 0.0589097
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0007579, 0.0034690, -0.0199498, 0.0029588, -0.0036276, 0.0233627
1: -0.0022277, 0.0061595, -0.0131660, 0.0056035, -0.0078312, 0.0191052
2: 0.0081640, 0.0207842, 0.0025143, 0.0220780, -0.0139140, 0.0182699
3: -0.0090497, 0.0041688, -0.0065645, 0.0128061, -0.0218558, 0.0107334
4: -0.0085193, 0.0058492, -0.0082663, 0.0147601, -0.0227377, 0.0141155
5: -0.0084388, 0.0120994, -0.0032530, 0.0310294, -0.0394682, 0.0153524
6: 0.0010856, 0.0116278, -0.0031414, 0.0115014, -0.0104158, 0.0145535
7: -0.0246659, -0.0039918, -0.0354255, -0.0039602, -0.0207057, 0.0310912
8: 0.9531202, 1.0113971, 0.9373071, 1.0074999, -0.0543798, 0.0740901
9: -0.0043646, 0.0148753, -0.0037525, 0.0504662, -0.0536180, 0.0186277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=8, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0472746, upper bound: 0.0484627
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0495432, upper bound: 0.0523627
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0009509, 0.0033333, -0.0236739, 0.0029816, -0.0038107, 0.0269267
1: -0.0026453, 0.0060253, -0.0152183, 0.0056360, -0.0082813, 0.0210218
2: 0.0082267, 0.0211768, -0.0009983, 0.0223155, -0.0140888, 0.0221751
3: -0.0082397, 0.0046515, -0.0066709, 0.0144476, -0.0226873, 0.0113225
4: -0.0084758, 0.0050352, -0.0088817, 0.0172179, -0.0251891, 0.0139169
5: -0.0065462, 0.0125687, -0.0036755, 0.0345643, -0.0411106, 0.0162443
6: 0.0015161, 0.0116100, -0.0051574, 0.0115082, -0.0099921, 0.0165929
7: -0.0256848, -0.0043494, -0.0372760, -0.0001404, -0.0255444, 0.0325463
8: 0.9502010, 1.0104990, 0.9348087, 1.0076886, -0.0574876, 0.0756903
9: -0.0042786, 0.0150200, -0.0037858, 0.0577528, -0.0609772, 0.0188058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0529862, upper bound: 0.0488707
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0529862, upper bound: 0.0571431
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0008691, 0.0035544, -0.0025234, 0.0033381, -0.0041485, 0.0060499
1: -0.0025954, 0.0062635, -0.0035247, 0.0060483, -0.0086437, 0.0097882
2: 0.0080485, 0.0212971, 0.0081748, 0.0213240, -0.0132755, 0.0131223
3: -0.0093220, 0.0045868, -0.0081479, 0.0051031, -0.0144251, 0.0127347
4: -0.0085994, 0.0062467, -0.0085062, 0.0057377, -0.0143371, 0.0147529
5: -0.0090370, 0.0125127, -0.0063107, 0.0144226, -0.0234596, 0.0188234
6: 0.0008334, 0.0116605, 0.0014416, 0.0116210, -0.0107876, 0.0102189
7: -0.0255630, -0.0037287, -0.0267322, -0.0043014, -0.0212616, 0.0230034
8: 0.9505497, 1.0121034, 0.9490441, 1.0106333, -0.0600836, 0.0630593
9: -0.0045231, 0.0155630, -0.0043319, 0.0174040, -0.0215615, 0.0198949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0533195, upper bound: 0.0511223
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0572520, upper bound: 0.0588564
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0026798, 0.0032166, -0.0025234, 0.0033381, -0.0059661, 0.0056595
1: -0.0034149, 0.0058919, -0.0035247, 0.0060483, -0.0094632, 0.0094167
2: 0.0084479, 0.0221428, 0.0081748, 0.0213240, -0.0128761, 0.0139681
3: -0.0077144, 0.0057066, -0.0081479, 0.0051031, -0.0128175, 0.0138545
4: -0.0083224, 0.0048962, -0.0085062, 0.0057377, -0.0140601, 0.0134024
5: -0.0054995, 0.0134339, -0.0063107, 0.0144226, -0.0199221, 0.0197446
6: 0.0018522, 0.0115473, 0.0014416, 0.0116210, -0.0097688, 0.0101057
7: -0.0275629, -0.0047229, -0.0267322, -0.0043014, -0.0232614, 0.0220093
8: 0.9448199, 1.0093627, 0.9490441, 1.0106333, -0.0658135, 0.0603186
9: -0.0039751, 0.0160963, -0.0043319, 0.0174040, -0.0213791, 0.0204283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0533195, upper bound: 0.0511223
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0572520, upper bound: 0.0588564
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011786, 0.0030851, -0.0218375, 0.0028292, -0.0039202, 0.0248727
1: -0.0027462, 0.0056435, -0.0142117, 0.0054315, -0.0081776, 0.0197906
2: 0.0087948, 0.0213276, 0.0007292, 0.0221427, -0.0133479, 0.0205984
3: -0.0078247, 0.0047899, -0.0061411, 0.0136413, -0.0214660, 0.0109310
4: -0.0080818, 0.0050254, -0.0081513, 0.0157891, -0.0234997, 0.0131767
5: -0.0061410, 0.0126821, -0.0024721, 0.0328305, -0.0389716, 0.0151543
6: 0.0020431, 0.0114491, -0.0040651, 0.0114461, -0.0094030, 0.0154828
7: -0.0259309, -0.0052865, -0.0363684, -0.0022459, -0.0236850, 0.0310101
8: 0.9494958, 1.0077974, 0.9360341, 1.0063236, -0.0568278, 0.0717633
9: -0.0034992, 0.0152560, -0.0034845, 0.0540446, -0.0567525, 0.0187405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0533491, upper bound: 0.0539762
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0532913, upper bound: 0.0538005
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0008618, 0.0035044, -0.0219795, 0.0029728, -0.0037578, 0.0254342
1: -0.0025923, 0.0061823, -0.0142844, 0.0056232, -0.0082155, 0.0202718
2: 0.0081748, 0.0212892, 0.0006000, 0.0222078, -0.0140331, 0.0206892
3: -0.0092483, 0.0045825, -0.0066298, 0.0137007, -0.0229490, 0.0112122
4: -0.0085118, 0.0062484, -0.0082834, 0.0161070, -0.0240579, 0.0145318
5: -0.0089636, 0.0125091, -0.0035046, 0.0329559, -0.0419195, 0.0160138
6: 0.0009438, 0.0116247, -0.0042426, 0.0115057, -0.0105619, 0.0156469
7: -0.0255554, -0.0039237, -0.0364340, -0.0019524, -0.0236030, 0.0322361
8: 0.9505717, 1.0115283, 0.9359456, 1.0076159, -0.0570443, 0.0755826
9: -0.0043499, 0.0155538, -0.0037734, 0.0544387, -0.0575687, 0.0193272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0525627, upper bound: 0.0487430
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0570115, upper bound: 0.0570115
time: 1.37 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0008296, 0.0032968, -0.0036629, 0.0039282, -0.0046905, 0.0068768
1: -0.0025635, 0.0059931, -0.0041302, 0.0069449, -0.0095084, 0.0101233
2: 0.0082290, 0.0210260, 0.0068443, 0.0216100, -0.0133810, 0.0141817
3: -0.0079726, 0.0045469, -0.0092830, 0.0055923, -0.0135649, 0.0138300
4: -0.0084742, 0.0046795, -0.0094270, 0.0070838, -0.0155580, 0.0141065
5: -0.0059143, 0.0124768, -0.0077622, 0.0154656, -0.0213799, 0.0202390
6: 0.0016675, 0.0116094, 0.0000494, 0.0119952, -0.0103277, 0.0115599
7: -0.0254852, -0.0044353, -0.0272782, -0.0021091, -0.0233761, 0.0226775
8: 0.9507726, 1.0102998, 0.9483070, 1.0168808, -0.0661082, 0.0619928
9: -0.0042755, 0.0147465, -0.0061448, 0.0200916, -0.0239188, 0.0208913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0522276, upper bound: 0.0485085
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0522276, upper bound: 0.0485085
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010050, 0.0029286, -0.0034950, 0.0037602, -0.0046878, 0.0063252
1: -0.0033846, 0.0055435, -0.0040518, 0.0067507, -0.0101353, 0.0095952
2: 0.0086365, 0.0218859, 0.0069947, 0.0214537, -0.0128171, 0.0148912
3: -0.0063726, 0.0054716, -0.0084843, 0.0055265, -0.0118990, 0.0139560
4: -0.0081915, 0.0033069, -0.0093241, 0.0061617, -0.0143533, 0.0126310
5: -0.0020846, 0.0133998, -0.0059599, 0.0153304, -0.0174150, 0.0193597
6: 0.0027484, 0.0114939, 0.0005856, 0.0119539, -0.0092054, 0.0109083
7: -0.0274890, -0.0055842, -0.0272074, -0.0026112, -0.0248777, 0.0215877
8: 0.9450317, 1.0072473, 0.9484025, 1.0155829, -0.0705512, 0.0588448
9: -0.0037163, 0.0153011, -0.0059444, 0.0193935, -0.0229653, 0.0212437

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0522276, upper bound: 0.0485085
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0522276, upper bound: 0.0485085
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0014686, 0.0033161, -0.0156272, 0.0037840, -0.0051478, 0.0188249
1: -0.0029347, 0.0060212, -0.0094602, 0.0068012, -0.0097359, 0.0152549
2: 0.0082038, 0.0212416, 0.0068968, 0.0276613, -0.0188123, 0.0143448
3: -0.0080429, 0.0047469, -0.0088205, 0.0098502, -0.0178931, 0.0135674
4: -0.0084896, 0.0050842, -0.0094760, 0.0118670, -0.0203469, 0.0145602
5: -0.0060643, 0.0132532, -0.0065979, 0.0263846, -0.0324489, 0.0198511
6: 0.0015789, 0.0116156, -0.0023648, 0.0119596, -0.0103806, 0.0139805
7: -0.0261200, -0.0043699, -0.0320840, -0.0019851, -0.0241349, 0.0270337
8: 0.9496977, 1.0104638, 0.9400799, 1.0157975, -0.0660998, 0.0703839
9: -0.0043059, 0.0155469, -0.0060772, 0.0382619, -0.0413045, 0.0216241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0573626, upper bound: 0.0577606
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0573626, upper bound: 0.0577606
time: 1.40 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0139164, 0.0029510, -0.0154003, 0.0036167, -0.0173978, 0.0182168
1: -0.0101451, 0.0055963, -0.0093764, 0.0066017, -0.0167468, 0.0149727
2: 0.0085520, 0.0221228, 0.0070570, 0.0274163, -0.0188643, 0.0150658
3: -0.0065285, 0.0058755, -0.0079961, 0.0097803, -0.0163088, 0.0138716
4: -0.0082081, 0.0152729, -0.0093498, 0.0111668, -0.0193749, 0.0246227
5: -0.0023325, 0.0318078, -0.0047573, 0.0262025, -0.0285350, 0.0365651
6: -0.0004441, 0.0115007, -0.0019334, 0.0119191, -0.0123632, 0.0134341
7: -0.0358330, -0.0054864, -0.0320085, -0.0026221, -0.0332109, 0.0265221
8: 0.9435127, 1.0074497, 0.9402196, 1.0144941, -0.0709814, 0.0672301
9: -0.0037491, 0.0317911, -0.0058612, 0.0376165, -0.0413656, 0.0376523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0548631, upper bound: 0.0548530
time: 1.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0572103, upper bound: 0.0576434
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0008020, 0.0035308, -0.0020343, 0.0039188, -0.0046640, 0.0055481
1: -0.0024341, 0.0062362, -0.0032318, 0.0069311, -0.0093652, 0.0094680
2: 0.0080695, 0.0210688, 0.0068688, 0.0215135, -0.0134439, 0.0142000
3: -0.0092309, 0.0044012, -0.0092353, 0.0048739, -0.0141048, 0.0136365
4: -0.0085848, 0.0060685, -0.0094138, 0.0062575, -0.0148423, 0.0154823
5: -0.0088306, 0.0123313, -0.0076034, 0.0139181, -0.0227487, 0.0199348
6: 0.0009134, 0.0116545, 0.0001760, 0.0119927, -0.0110793, 0.0114785
7: -0.0251694, -0.0037980, -0.0264681, -0.0021437, -0.0230257, 0.0226701
8: 0.9516774, 1.0119243, 0.9494008, 1.0168037, -0.0651263, 0.0625235
9: -0.0044943, 0.0152599, -0.0061323, 0.0170046, -0.0214989, 0.0213922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0519943, upper bound: 0.0484720
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0519943, upper bound: 0.0484720
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0009777, 0.0031937, -0.0018674, 0.0037511, -0.0046645, 0.0050141
1: -0.0032566, 0.0058287, -0.0031541, 0.0067376, -0.0099942, 0.0089829
2: 0.0084718, 0.0219067, 0.0070169, 0.0213551, -0.0128833, 0.0148898
3: -0.0076295, 0.0053275, -0.0084401, 0.0048086, -0.0124382, 0.0137676
4: -0.0083058, 0.0047198, -0.0093119, 0.0053099, -0.0136157, 0.0140317
5: -0.0052246, 0.0132559, -0.0058086, 0.0137843, -0.0190088, 0.0190646
6: 0.0019262, 0.0115406, 0.0006963, 0.0119513, -0.0100251, 0.0108443
7: -0.0271766, -0.0048334, -0.0263980, -0.0026433, -0.0245332, 0.0215646
8: 0.9459267, 1.0091459, 0.9494953, 1.0155082, -0.0695815, 0.0596506
9: -0.0039423, 0.0157674, -0.0059320, 0.0162818, -0.0202242, 0.0216993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0509096, upper bound: 0.0477142
time: 1.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0501860, upper bound: 0.0465268
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0008691, 0.0035544, -0.0135553, 0.0037748, -0.0045645, 0.0170554
1: -0.0025954, 0.0062635, -0.0086040, 0.0067832, -0.0093786, 0.0147693
2: 0.0080485, 0.0212971, 0.0069190, 0.0263077, -0.0176983, 0.0143781
3: -0.0093220, 0.0045868, -0.0087250, 0.0091655, -0.0184875, 0.0133118
4: -0.0085994, 0.0062467, -0.0094407, 0.0109007, -0.0195001, 0.0156874
5: -0.0090370, 0.0125127, -0.0064435, 0.0245232, -0.0335603, 0.0189562
6: 0.0008334, 0.0116605, -0.0012795, 0.0119568, -0.0111234, 0.0129400
7: -0.0255630, -0.0037287, -0.0313120, -0.0023997, -0.0231633, 0.0272730
8: 0.9505497, 1.0121034, 0.9415089, 1.0157213, -0.0651716, 0.0705945
9: -0.0045231, 0.0155630, -0.0060404, 0.0352478, -0.0385114, 0.0216033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0568946, upper bound: 0.0577309
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0568946, upper bound: 0.0577309
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0026798, 0.0032166, -0.0133282, 0.0036077, -0.0061925, 0.0164580
1: -0.0034149, 0.0058919, -0.0085203, 0.0065865, -0.0100015, 0.0144122
2: 0.0084479, 0.0221428, 0.0070772, 0.0260574, -0.0175324, 0.0150657
3: -0.0077144, 0.0057066, -0.0079062, 0.0090957, -0.0168101, 0.0136128
4: -0.0083224, 0.0048962, -0.0093193, 0.0101889, -0.0185113, 0.0142155
5: -0.0054995, 0.0134339, -0.0046098, 0.0243413, -0.0298408, 0.0180437
6: 0.0018522, 0.0115473, -0.0007770, 0.0119163, -0.0100641, 0.0123243
7: -0.0275629, -0.0047229, -0.0312366, -0.0029783, -0.0245845, 0.0265137
8: 0.9448199, 1.0093627, 0.9416486, 1.0144192, -0.0695993, 0.0677141
9: -0.0039751, 0.0160963, -0.0058283, 0.0345961, -0.0385712, 0.0219247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0545435, upper bound: 0.0548173
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0567661, upper bound: 0.0576114
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010690, 0.0038966, -0.0010394, 0.0033285, -0.0043391, 0.0048639
1: -0.0026827, 0.0069037, -0.0026865, 0.0060329, -0.0087156, 0.0095903
2: 0.0068935, 0.0214145, 0.0082007, 0.0212124, -0.0143188, 0.0132138
3: -0.0091286, 0.0047069, -0.0080978, 0.0047076, -0.0138362, 0.0128047
4: -0.0094005, 0.0055321, -0.0084938, 0.0048703, -0.0142708, 0.0140259
5: -0.0073826, 0.0126108, -0.0061906, 0.0126151, -0.0199977, 0.0188014
6: 0.0002869, 0.0119876, 0.0015716, 0.0116173, -0.0113304, 0.0104160
7: -0.0257761, -0.0022118, -0.0257854, -0.0043371, -0.0214390, 0.0234630
8: 0.9499394, 1.0166347, 0.9499125, 1.0105482, -0.0606089, 0.0667222
9: -0.0061078, 0.0155770, -0.0043143, 0.0150107, -0.0210315, 0.0198913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0441505, upper bound: 0.0472586
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0504925, upper bound: 0.0535316
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0115032, 0.0037533, -0.0042159, 0.0033481, -0.0147361, 0.0078525
1: -0.0077566, 0.0067545, -0.0044578, 0.0060629, -0.0137365, 0.0112123
2: 0.0069502, 0.0249545, 0.0081476, 0.0214286, -0.0144784, 0.0168069
3: -0.0085695, 0.0084877, -0.0081981, 0.0058493, -0.0144188, 0.0166858
4: -0.0093989, 0.0099085, -0.0085208, 0.0065746, -0.0159735, 0.0184293
5: -0.0061532, 0.0226809, -0.0064767, 0.0160298, -0.0221830, 0.0291576
6: -0.0003981, 0.0119517, 0.0012915, 0.0116237, -0.0120219, 0.0106602
7: -0.0305480, -0.0025912, -0.0275736, -0.0042653, -0.0262696, 0.0249823
8: 0.9429232, 1.0155566, 0.9479083, 1.0107148, -0.0677916, 0.0676483
9: -0.0059918, 0.0322244, -0.0043452, 0.0206394, -0.0266313, 0.0354752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0540103, upper bound: 0.0510472
time: 1.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0540103, upper bound: 0.0585431
time: 1.47 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010291, 0.0037296, -0.0199498, 0.0029588, -0.0039125, 0.0235620
1: -0.0026767, 0.0067113, -0.0131660, 0.0056035, -0.0082802, 0.0196126
2: 0.0070395, 0.0212549, 0.0025143, 0.0220780, -0.0150386, 0.0187406
3: -0.0083402, 0.0046956, -0.0065645, 0.0128061, -0.0211463, 0.0112601
4: -0.0092993, 0.0046208, -0.0082663, 0.0147601, -0.0235918, 0.0128871
5: -0.0056008, 0.0126041, -0.0032530, 0.0310294, -0.0366301, 0.0158571
6: 0.0007981, 0.0119463, -0.0031414, 0.0115014, -0.0107032, 0.0149630
7: -0.0257615, -0.0027087, -0.0354255, -0.0039602, -0.0218013, 0.0321850
8: 0.9499810, 1.0153438, 0.9373071, 1.0074999, -0.0575189, 0.0780367
9: -0.0059076, 0.0150903, -0.0037525, 0.0504662, -0.0553086, 0.0188427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=8, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0460622, upper bound: 0.0465607
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0494848, upper bound: 0.0518683
time: 1.20 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0112739, 0.0035868, -0.0236739, 0.0029816, -0.0141229, 0.0271249
1: -0.0076721, 0.0065591, -0.0152183, 0.0056360, -0.0133081, 0.0217774
2: 0.0071062, 0.0246985, -0.0009983, 0.0223155, -0.0152094, 0.0256968
3: -0.0077667, 0.0084172, -0.0066709, 0.0144476, -0.0222143, 0.0150881
4: -0.0092824, 0.0091730, -0.0088817, 0.0172179, -0.0265003, 0.0180546
5: -0.0043286, 0.0224972, -0.0036755, 0.0345643, -0.0388930, 0.0261727
6: 0.0001713, 0.0119113, -0.0051574, 0.0115082, -0.0113369, 0.0170687
7: -0.0304718, -0.0031031, -0.0372760, -0.0001404, -0.0303314, 0.0341730
8: 0.9430643, 1.0142572, 0.9348087, 1.0076886, -0.0646244, 0.0794485
9: -0.0057839, 0.0315646, -0.0037858, 0.0577528, -0.0635367, 0.0353503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0531947, upper bound: 0.0484279
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0531947, upper bound: 0.0569096
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0104882, 0.0038080, -0.0025234, 0.0033381, -0.0137547, 0.0062505
1: -0.0073317, 0.0068137, -0.0035247, 0.0060483, -0.0133800, 0.0103384
2: 0.0069121, 0.0243535, 0.0081748, 0.0213240, -0.0144119, 0.0161788
3: -0.0088798, 0.0081496, -0.0081479, 0.0051031, -0.0139829, 0.0162975
4: -0.0094169, 0.0098057, -0.0085062, 0.0057377, -0.0151546, 0.0183119
5: -0.0069544, 0.0217571, -0.0063107, 0.0144226, -0.0213770, 0.0280678
6: -0.0003667, 0.0119633, 0.0014416, 0.0116210, -0.0119877, 0.0105217
7: -0.0301648, -0.0024417, -0.0267322, -0.0043014, -0.0258634, 0.0242905
8: 0.9436325, 1.0159341, 0.9490441, 1.0106333, -0.0670009, 0.0668900
9: -0.0060386, 0.0309371, -0.0043319, 0.0174040, -0.0234426, 0.0345316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0537036, upper bound: 0.0508211
time: 1.07 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0578922, upper bound: 0.0585239
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0360842, 0.0034687, -0.0025234, 0.0033381, -0.0393513, 0.0058761
1: -0.0179331, 0.0149810, -0.0035247, 0.0060483, -0.0239035, 0.0185058
2: -0.0056495, 0.0409748, 0.0081748, 0.0213240, -0.0269735, 0.0326673
3: -0.0081812, 0.0166201, -0.0081479, 0.0051031, -0.0132843, 0.0247681
4: -0.0254870, 0.0207153, -0.0085062, 0.0057377, -0.0312247, 0.0289657
5: -0.0045041, 0.0448058, -0.0063107, 0.0144226, -0.0189267, 0.0511165
6: -0.0146179, 0.0118493, 0.0014416, 0.0116210, -0.0261864, 0.0104077
7: -0.0397239, 0.0272521, -0.0267322, -0.0043014, -0.0354012, 0.0539843
8: 0.9259380, 1.0131009, 0.9490441, 1.0106333, -0.0846953, 0.0640568
9: -0.0056790, 0.0675555, -0.0043319, 0.0174040, -0.0230830, 0.0711752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0537036, upper bound: 0.0508211
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0578922, upper bound: 0.0585239
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0008287, 0.0039388, -0.0182537, 0.0029501, -0.0037144, 0.0221317
1: -0.0025592, 0.0069437, -0.0122309, 0.0055908, -0.0081500, 0.0190043
2: 0.0068715, 0.0213402, 0.0041144, 0.0219693, -0.0150977, 0.0172257
3: -0.0094903, 0.0045421, -0.0065237, 0.0120583, -0.0215486, 0.0110658
4: -0.0094158, 0.0059673, -0.0082548, 0.0136519, -0.0225994, 0.0142221
5: -0.0083329, 0.0124720, -0.0030957, 0.0294187, -0.0377516, 0.0155677
6: 0.0001046, 0.0119938, -0.0022306, 0.0114989, -0.0113942, 0.0140580
7: -0.0254748, -0.0020993, -0.0345823, -0.0048911, -0.0205837, 0.0322475
8: 0.9508027, 1.0169210, 0.9384454, 1.0074285, -0.0566258, 0.0784756
9: -0.0061380, 0.0156631, -0.0037404, 0.0471408, -0.0521892, 0.0194035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0446023, upper bound: 0.0455401
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0477959, upper bound: 0.0513688
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0081694, 0.0038125, -0.0219795, 0.0029728, -0.0110203, 0.0257188
1: -0.0063727, 0.0068106, -0.0142844, 0.0056232, -0.0119959, 0.0210950
2: 0.0069365, 0.0228600, 0.0006000, 0.0222078, -0.0152714, 0.0222600
3: -0.0088896, 0.0073829, -0.0066298, 0.0137007, -0.0225902, 0.0140126
4: -0.0093771, 0.0088262, -0.0082834, 0.0161070, -0.0254841, 0.0171096
5: -0.0070434, 0.0196721, -0.0035046, 0.0329559, -0.0399993, 0.0231767
6: -0.0000295, 0.0119601, -0.0042426, 0.0115057, -0.0115352, 0.0162027
7: -0.0293001, -0.0024522, -0.0364340, -0.0019524, -0.0273478, 0.0339818
8: 0.9452331, 1.0159364, 0.9359456, 1.0076159, -0.0623828, 0.0799907
9: -0.0059963, 0.0276116, -0.0037734, 0.0544387, -0.0602328, 0.0313850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0528937, upper bound: 0.0484009
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0528937, upper bound: 0.0568946
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0150533, 0.0032585, -0.0035298, 0.0037855, -0.0185944, 0.0065059
1: -0.0092427, 0.0061013, -0.0040643, 0.0067582, -0.0152209, 0.0101656
2: 0.0076929, 0.0270878, 0.0070531, 0.0215228, -0.0138299, 0.0190872
3: -0.0069162, 0.0096705, -0.0087742, 0.0055379, -0.0124542, 0.0183761
4: -0.0089405, 0.0104396, -0.0092830, 0.0065915, -0.0155320, 0.0191139
5: -0.0026907, 0.0259120, -0.0067387, 0.0153520, -0.0180427, 0.0326506
6: -0.0014322, 0.0117425, 0.0004534, 0.0119368, -0.0133215, 0.0112891
7: -0.0318880, -0.0039201, -0.0272187, -0.0025781, -0.0276024, 0.0232987
8: 0.9404429, 1.0110722, 0.9483871, 1.0156038, -0.0727708, 0.0626851
9: -0.0049913, 0.0367948, -0.0058618, 0.0196363, -0.0246276, 0.0402983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0583594, upper bound: 0.0534025
time: 1.28 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0583594, upper bound: 0.0534025
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0091436, 0.0036580, -0.0036629, 0.0039282, -0.0128461, 0.0070404
1: -0.0067847, 0.0066263, -0.0041302, 0.0069449, -0.0132067, 0.0107565
2: 0.0071044, 0.0233770, 0.0068443, 0.0216100, -0.0145056, 0.0162693
3: -0.0081619, 0.0077097, -0.0092830, 0.0055923, -0.0137541, 0.0169927
4: -0.0092689, 0.0086054, -0.0094270, 0.0070838, -0.0163527, 0.0179951
5: -0.0053312, 0.0205680, -0.0077622, 0.0154656, -0.0207967, 0.0283302
6: 0.0003671, 0.0119126, 0.0000494, 0.0119952, -0.0116281, 0.0118632
7: -0.0296717, -0.0029269, -0.0272782, -0.0021091, -0.0260679, 0.0243513
8: 0.9445453, 1.0147034, 0.9483070, 1.0168808, -0.0718992, 0.0663964
9: -0.0057748, 0.0286851, -0.0061448, 0.0200916, -0.0248487, 0.0326227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0573490, upper bound: 0.0522614
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0573490, upper bound: 0.0557948
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010690, 0.0038966, -0.0156272, 0.0037840, -0.0046289, 0.0192974
1: -0.0026827, 0.0069037, -0.0094602, 0.0068012, -0.0094839, 0.0156085
2: 0.0068935, 0.0214145, 0.0068968, 0.0276613, -0.0192084, 0.0145176
3: -0.0091286, 0.0047069, -0.0088205, 0.0098502, -0.0189788, 0.0135274
4: -0.0094005, 0.0055321, -0.0094760, 0.0118670, -0.0206632, 0.0150081
5: -0.0073826, 0.0126108, -0.0065979, 0.0263846, -0.0337672, 0.0192087
6: 0.0002869, 0.0119876, -0.0023648, 0.0119596, -0.0116726, 0.0143525
7: -0.0257761, -0.0022118, -0.0320840, -0.0019851, -0.0237909, 0.0279275
8: 0.9499394, 1.0166347, 0.9400799, 1.0157975, -0.0658581, 0.0754925
9: -0.0061078, 0.0155770, -0.0060772, 0.0382619, -0.0419723, 0.0216542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0503998, upper bound: 0.0536631
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0497658, upper bound: 0.0524888
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0115032, 0.0037533, -0.0156272, 0.0037840, -0.0150150, 0.0191046
1: -0.0077566, 0.0067545, -0.0094602, 0.0068012, -0.0145578, 0.0162147
2: 0.0069502, 0.0249545, 0.0068968, 0.0276613, -0.0207112, 0.0180577
3: -0.0085695, 0.0084877, -0.0088205, 0.0098502, -0.0184197, 0.0173082
4: -0.0093989, 0.0099085, -0.0094760, 0.0118670, -0.0212659, 0.0193844
5: -0.0061532, 0.0226809, -0.0065979, 0.0263846, -0.0325378, 0.0292787
6: -0.0003981, 0.0119517, -0.0023648, 0.0119596, -0.0123577, 0.0143166
7: -0.0305480, -0.0025912, -0.0320840, -0.0019851, -0.0285628, 0.0294928
8: 0.9429232, 1.0155566, 0.9400799, 1.0157975, -0.0715577, 0.0735512
9: -0.0059918, 0.0322244, -0.0060772, 0.0382619, -0.0429299, 0.0378395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0503998, upper bound: 0.0584416
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0497658, upper bound: 0.0569583
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0058113, 0.0037854, -0.0020343, 0.0039188, -0.0095146, 0.0055978
1: -0.0053268, 0.0067786, -0.0032318, 0.0069311, -0.0120340, 0.0100104
2: 0.0069640, 0.0216234, 0.0068688, 0.0215135, -0.0145494, 0.0145429
3: -0.0086637, 0.0065466, -0.0092353, 0.0048739, -0.0135376, 0.0157819
4: -0.0093407, 0.0076030, -0.0094138, 0.0062575, -0.0155982, 0.0170168
5: -0.0065306, 0.0175266, -0.0076034, 0.0139181, -0.0204487, 0.0251301
6: 0.0003315, 0.0119571, 0.0001760, 0.0119927, -0.0116611, 0.0117811
7: -0.0283571, -0.0025355, -0.0264681, -0.0021437, -0.0245130, 0.0236682
8: 0.9468502, 1.0157501, 0.9494008, 1.0168037, -0.0699535, 0.0663493
9: -0.0059601, 0.0239171, -0.0061323, 0.0170046, -0.0220263, 0.0280128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0526644, upper bound: 0.0482632
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0526644, upper bound: 0.0482632
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0250205, 0.0034468, -0.0018674, 0.0037511, -0.0285321, 0.0050902
1: -0.0159546, 0.0063761, -0.0031541, 0.0067376, -0.0219037, 0.0095303
2: -0.0022634, 0.0224661, 0.0070169, 0.0213551, -0.0236185, 0.0152310
3: -0.0072820, 0.0150377, -0.0084401, 0.0048086, -0.0120906, 0.0234369
4: -0.0094190, 0.0183330, -0.0093119, 0.0053099, -0.0147289, 0.0266167
5: -0.0040647, 0.0358327, -0.0058086, 0.0137843, -0.0178490, 0.0416413
6: -0.0060125, 0.0118427, 0.0006963, 0.0119513, -0.0176557, 0.0111464
7: -0.0379400, 0.0014826, -0.0263980, -0.0026433, -0.0334273, 0.0278806
8: 0.9339125, 1.0129187, 0.9494953, 1.0155082, -0.0800788, 0.0634235
9: -0.0054057, 0.0605253, -0.0059320, 0.0162818, -0.0216875, 0.0643038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0513067, upper bound: 0.0475455
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0505475, upper bound: 0.0463174
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0104882, 0.0038080, -0.0135553, 0.0037748, -0.0140296, 0.0171169
1: -0.0073317, 0.0068137, -0.0086040, 0.0067832, -0.0141149, 0.0154177
2: 0.0069121, 0.0243535, 0.0069190, 0.0263077, -0.0193956, 0.0174345
3: -0.0088798, 0.0081496, -0.0087250, 0.0091655, -0.0180453, 0.0168745
4: -0.0094169, 0.0098057, -0.0094407, 0.0109007, -0.0203176, 0.0192464
5: -0.0069544, 0.0217571, -0.0064435, 0.0245232, -0.0314776, 0.0282006
6: -0.0003667, 0.0119633, -0.0012795, 0.0119568, -0.0123235, 0.0132428
7: -0.0301648, -0.0024417, -0.0313120, -0.0023997, -0.0277651, 0.0288703
8: 0.9436325, 1.0159341, 0.9415089, 1.0157213, -0.0720888, 0.0733463
9: -0.0060386, 0.0309371, -0.0060404, 0.0352478, -0.0398838, 0.0365258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0577299, upper bound: 0.0572011
time: 1.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0577299, upper bound: 0.0572011
time: 1.56 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0360842, 0.0034687, -0.0133282, 0.0036077, -0.0394393, 0.0165441
1: -0.0179331, 0.0149810, -0.0085203, 0.0065865, -0.0245196, 0.0235013
2: -0.0056495, 0.0409748, 0.0070772, 0.0260574, -0.0317069, 0.0338977
3: -0.0081812, 0.0166201, -0.0079062, 0.0090957, -0.0172768, 0.0245263
4: -0.0254870, 0.0207153, -0.0093193, 0.0101889, -0.0356759, 0.0300346
5: -0.0045041, 0.0448058, -0.0046098, 0.0243413, -0.0288454, 0.0494156
6: -0.0146179, 0.0118493, -0.0007770, 0.0119163, -0.0265342, 0.0126263
7: -0.0397239, 0.0272521, -0.0312366, -0.0029783, -0.0367456, 0.0584888
8: 0.9259380, 1.0131009, 0.9416486, 1.0144192, -0.0858444, 0.0710561
9: -0.0056790, 0.0675555, -0.0058283, 0.0345961, -0.0402751, 0.0729472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0550688, upper bound: 0.0545171
time: 1.58 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0576106, upper bound: 0.0570268
time: 1.48 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.50 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0536679, upper bound: 0.0513409
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0577316, upper bound: 0.0589097
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0536679, upper bound: 0.0513409
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0577316, upper bound: 0.0589097
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0472746, upper bound: 0.0484627
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0495432, upper bound: 0.0523627
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0529862, upper bound: 0.0488707
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0529862, upper bound: 0.0571431
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0533195, upper bound: 0.0511223
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0572520, upper bound: 0.0588564
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0533195, upper bound: 0.0511223
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0572520, upper bound: 0.0588564
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0533491, upper bound: 0.0539762
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0532913, upper bound: 0.0538005
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0525627, upper bound: 0.0487430
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0570115, upper bound: 0.0570115
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0522276, upper bound: 0.0485085
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0522276, upper bound: 0.0485085
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0522276, upper bound: 0.0485085
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0522276, upper bound: 0.0485085
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0573626, upper bound: 0.0577606
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0573626, upper bound: 0.0577606
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0548631, upper bound: 0.0548530
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0572103, upper bound: 0.0576434
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0519943, upper bound: 0.0484720
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0519943, upper bound: 0.0484720
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0509096, upper bound: 0.0477142
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0501860, upper bound: 0.0465268
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0568946, upper bound: 0.0577309
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0568946, upper bound: 0.0577309
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0545435, upper bound: 0.0548173
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0567661, upper bound: 0.0576114
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0441505, upper bound: 0.0472586
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0504925, upper bound: 0.0535316
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0540103, upper bound: 0.0510472
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0540103, upper bound: 0.0585431
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0460622, upper bound: 0.0465607
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0494848, upper bound: 0.0518683
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0531947, upper bound: 0.0484279
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0531947, upper bound: 0.0569096
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0537036, upper bound: 0.0508211
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0578922, upper bound: 0.0585239
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0537036, upper bound: 0.0508211
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0578922, upper bound: 0.0585239
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0446023, upper bound: 0.0455401
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0477959, upper bound: 0.0513688
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0528937, upper bound: 0.0484009
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0528937, upper bound: 0.0568946
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0583594, upper bound: 0.0534025
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0583594, upper bound: 0.0534025
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0573490, upper bound: 0.0522614
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0573490, upper bound: 0.0557948
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0503998, upper bound: 0.0536631
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0497658, upper bound: 0.0524888
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0503998, upper bound: 0.0584416
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0497658, upper bound: 0.0569583
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0526644, upper bound: 0.0482632
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0526644, upper bound: 0.0482632
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0513067, upper bound: 0.0475455
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0505475, upper bound: 0.0463174
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0577299, upper bound: 0.0572011
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0577299, upper bound: 0.0572011
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0550688, upper bound: 0.0545171
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 8, lower bound: -0.0576106, upper bound: 0.0570268

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0008296, 0.0032968, -0.0007828, 0.0034666, -0.0042503, 0.0040069
1: -0.0025635, 0.0059931, -0.0023444, 0.0061701, -0.0087336, 0.0083376
2: 0.0082290, 0.0210260, 0.0081278, 0.0209114, -0.0126825, 0.0128983
3: -0.0079726, 0.0045469, -0.0089043, 0.0043003, -0.0122728, 0.0134512
4: -0.0084742, 0.0046795, -0.0085444, 0.0057257, -0.0142000, 0.0132239
5: -0.0059143, 0.0124768, -0.0080932, 0.0122306, -0.0181449, 0.0205700
6: 0.0016675, 0.0116094, 0.0011283, 0.0116380, -0.0099705, 0.0104811
7: -0.0254852, -0.0044353, -0.0249506, -0.0039691, -0.0213735, 0.0203544
8: 0.9507726, 1.0102998, 0.9523044, 1.0114783, -0.0607057, 0.0579954
9: -0.0042755, 0.0147465, -0.0044144, 0.0149674, -0.0190816, 0.0188373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0563027, upper bound: 0.0522557
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0578783, upper bound: 0.0539938
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0014686, 0.0033161, -0.0025821, 0.0033348, -0.0047179, 0.0057725
1: -0.0029347, 0.0060212, -0.0035581, 0.0060450, -0.0089797, 0.0095792
2: 0.0082038, 0.0212416, 0.0081753, 0.0213194, -0.0131156, 0.0130663
3: -0.0080429, 0.0047469, -0.0081291, 0.0051295, -0.0131724, 0.0128760
4: -0.0084896, 0.0050842, -0.0085057, 0.0057372, -0.0142268, 0.0135899
5: -0.0060643, 0.0132532, -0.0062653, 0.0144801, -0.0205443, 0.0195186
6: 0.0015789, 0.0116156, 0.0014489, 0.0116207, -0.0100417, 0.0101667
7: -0.0261200, -0.0043699, -0.0267623, -0.0043102, -0.0218098, 0.0217612
8: 0.9496977, 1.0104638, 0.9490036, 1.0106113, -0.0609136, 0.0614603
9: -0.0043059, 0.0155469, -0.0043304, 0.0174938, -0.0210851, 0.0197785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0550480, upper bound: 0.0577187
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0550480, upper bound: 0.0613858
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0115490, 0.0029286, -0.0007828, 0.0034666, -0.0149437, 0.0036029
1: -0.0087810, 0.0055644, -0.0023444, 0.0061701, -0.0147606, 0.0079088
2: 0.0085873, 0.0218859, 0.0081278, 0.0209114, -0.0123241, 0.0137581
3: -0.0064244, 0.0056605, -0.0089043, 0.0043003, -0.0107246, 0.0145647
4: -0.0081915, 0.0128342, -0.0085444, 0.0057257, -0.0139173, 0.0208418
5: -0.0021379, 0.0282975, -0.0080932, 0.0122306, -0.0143685, 0.0363907
6: 0.0008389, 0.0114939, 0.0011283, 0.0116380, -0.0107991, 0.0103656
7: -0.0339955, -0.0055604, -0.0249506, -0.0039691, -0.0295664, 0.0193903
8: 0.9446828, 1.0072644, 0.9523044, 1.0114783, -0.0667955, 0.0549600
9: -0.0037163, 0.0284917, -0.0044144, 0.0149674, -0.0186837, 0.0316352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0501423, upper bound: 0.0488724
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0530275, upper bound: 0.0506864
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0139164, 0.0029510, -0.0025821, 0.0033348, -0.0171578, 0.0053716
1: -0.0101451, 0.0055963, -0.0035581, 0.0060450, -0.0160605, 0.0091543
2: 0.0085520, 0.0221228, 0.0081753, 0.0213194, -0.0127674, 0.0139475
3: -0.0065285, 0.0058755, -0.0081291, 0.0051295, -0.0116580, 0.0140046
4: -0.0082081, 0.0152729, -0.0085057, 0.0057372, -0.0139453, 0.0234268
5: -0.0023325, 0.0318078, -0.0062653, 0.0144801, -0.0168126, 0.0380732
6: -0.0004441, 0.0115007, 0.0014489, 0.0116207, -0.0120648, 0.0100518
7: -0.0358330, -0.0054864, -0.0267623, -0.0043102, -0.0312660, 0.0211859
8: 0.9435127, 1.0074497, 0.9490036, 1.0106113, -0.0670986, 0.0584462
9: -0.0037491, 0.0317911, -0.0043304, 0.0174938, -0.0212429, 0.0350978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0506780, upper bound: 0.0542632
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0506780, upper bound: 0.0589097
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0007562, 0.0034611, -0.0174452, 0.0028583, -0.0035139, 0.0208370
1: -0.0022195, 0.0061525, -0.0117968, 0.0055004, -0.0077199, 0.0176712
2: 0.0081653, 0.0207598, 0.0048672, 0.0218083, -0.0136430, 0.0158926
3: -0.0089997, 0.0041596, -0.0059947, 0.0117086, -0.0207083, 0.0101543
4: -0.0085184, 0.0057740, -0.0082386, 0.0126691, -0.0205780, 0.0140126
5: -0.0083133, 0.0120902, -0.0015148, 0.0286710, -0.0369843, 0.0136050
6: 0.0011143, 0.0116274, -0.0015983, 0.0114970, -0.0103827, 0.0130026
7: -0.0246459, -0.0040104, -0.0341909, -0.0054492, -0.0191967, 0.0296916
8: 0.9531776, 1.0113521, 0.9389739, 1.0068756, -0.0536981, 0.0723782
9: -0.0043629, 0.0148185, -0.0037310, 0.0453238, -0.0483120, 0.0185495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0487058, upper bound: 0.0515020
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0484687, upper bound: 0.0512766
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0009509, 0.0033333, -0.0116088, 0.0031206, -0.0039683, 0.0148605
1: -0.0026453, 0.0060253, -0.0085524, 0.0057737, -0.0084189, 0.0144596
2: 0.0082267, 0.0211768, 0.0084228, 0.0216977, -0.0134710, 0.0127540
3: -0.0082397, 0.0046515, -0.0073520, 0.0091198, -0.0173594, 0.0120035
4: -0.0084758, 0.0050352, -0.0083152, 0.0100457, -0.0183761, 0.0133504
5: -0.0065462, 0.0125687, -0.0048696, 0.0230826, -0.0296289, 0.0174384
6: 0.0015161, 0.0116100, 0.0006557, 0.0115281, -0.0100120, 0.0109543
7: -0.0256848, -0.0043494, -0.0312656, -0.0049782, -0.0207065, 0.0266256
8: 0.9502010, 1.0104990, 0.9429235, 1.0086730, -0.0584719, 0.0675755
9: -0.0042786, 0.0150200, -0.0038818, 0.0345003, -0.0377001, 0.0189019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0492170, upper bound: 0.0467462
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0523427, upper bound: 0.0482045
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0009509, 0.0033333, -0.0219949, 0.0029667, -0.0037947, 0.0252140
1: -0.0026453, 0.0060253, -0.0142934, 0.0056166, -0.0082619, 0.0199396
2: 0.0082267, 0.0211768, 0.0005850, 0.0222009, -0.0139742, 0.0205918
3: -0.0082397, 0.0046515, -0.0065984, 0.0137078, -0.0219474, 0.0112499
4: -0.0084758, 0.0050352, -0.0082808, 0.0160826, -0.0239006, 0.0133160
5: -0.0065462, 0.0125687, -0.0034207, 0.0329712, -0.0395174, 0.0159895
6: 0.0015161, 0.0116100, -0.0042360, 0.0115049, -0.0099888, 0.0156153
7: -0.0256848, -0.0043494, -0.0364420, -0.0019541, -0.0237306, 0.0313217
8: 0.9502010, 1.0104990, 0.9359347, 1.0075740, -0.0573729, 0.0745643
9: -0.0042786, 0.0150200, -0.0037695, 0.0544553, -0.0573243, 0.0187895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0492170, upper bound: 0.0539965
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0523427, upper bound: 0.0563308
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0008020, 0.0035308, -0.0007669, 0.0034569, -0.0042234, 0.0042881
1: -0.0024341, 0.0062362, -0.0022699, 0.0061588, -0.0085929, 0.0085061
2: 0.0080695, 0.0210688, 0.0081368, 0.0208096, -0.0127401, 0.0129320
3: -0.0092309, 0.0044012, -0.0088729, 0.0042163, -0.0134472, 0.0132741
4: -0.0085848, 0.0060685, -0.0085381, 0.0056638, -0.0142486, 0.0146067
5: -0.0088306, 0.0123313, -0.0080308, 0.0121468, -0.0209774, 0.0203622
6: 0.0009134, 0.0116545, 0.0011593, 0.0116355, -0.0107221, 0.0104952
7: -0.0251694, -0.0037980, -0.0247688, -0.0039984, -0.0211710, 0.0209708
8: 0.9516774, 1.0119243, 0.9528254, 1.0114015, -0.0597241, 0.0590989
9: -0.0044943, 0.0152599, -0.0044020, 0.0148420, -0.0192880, 0.0195431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0559736, upper bound: 0.0522528
time: 1.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0575259, upper bound: 0.0539796
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0008691, 0.0035544, -0.0012478, 0.0033248, -0.0041337, 0.0047516
1: -0.0025954, 0.0062635, -0.0028088, 0.0060305, -0.0086259, 0.0090723
2: 0.0080485, 0.0212971, 0.0081971, 0.0212170, -0.0131684, 0.0131000
3: -0.0093220, 0.0045868, -0.0080795, 0.0047268, -0.0140488, 0.0126663
4: -0.0085994, 0.0062467, -0.0084953, 0.0049816, -0.0135810, 0.0147420
5: -0.0090370, 0.0125127, -0.0061471, 0.0129294, -0.0219664, 0.0186598
6: 0.0008334, 0.0116605, 0.0015682, 0.0116180, -0.0107846, 0.0100922
7: -0.0255630, -0.0037287, -0.0259505, -0.0043448, -0.0212182, 0.0220467
8: 0.9505497, 1.0121034, 0.9498057, 1.0105300, -0.0599803, 0.0622978
9: -0.0045231, 0.0155630, -0.0043172, 0.0152547, -0.0193344, 0.0198801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0603857, upper bound: 0.0600598
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0612295, upper bound: 0.0612295
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0023172, 0.0031937, -0.0007669, 0.0034569, -0.0057420, 0.0038988
1: -0.0032566, 0.0058569, -0.0022699, 0.0061588, -0.0094154, 0.0081268
2: 0.0084718, 0.0219067, 0.0081368, 0.0208096, -0.0123378, 0.0137699
3: -0.0076295, 0.0054887, -0.0088729, 0.0042163, -0.0118458, 0.0143617
4: -0.0083058, 0.0047198, -0.0085381, 0.0056638, -0.0139696, 0.0132580
5: -0.0052900, 0.0132559, -0.0080308, 0.0121468, -0.0174368, 0.0212868
6: 0.0019262, 0.0115406, 0.0011593, 0.0116355, -0.0097093, 0.0103813
7: -0.0271766, -0.0048013, -0.0247688, -0.0039984, -0.0231782, 0.0199674
8: 0.9459267, 1.0091696, 0.9528254, 1.0114015, -0.0654749, 0.0563442
9: -0.0039423, 0.0157674, -0.0044020, 0.0148420, -0.0187844, 0.0200580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0490711, upper bound: 0.0482550
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0527025, upper bound: 0.0504740
time: 1.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0026798, 0.0032166, -0.0012478, 0.0033248, -0.0059513, 0.0043599
1: -0.0034149, 0.0058919, -0.0028088, 0.0060305, -0.0094455, 0.0087008
2: 0.0084479, 0.0221428, 0.0081971, 0.0212170, -0.0127690, 0.0139457
3: -0.0077144, 0.0057066, -0.0080795, 0.0047268, -0.0124412, 0.0137861
4: -0.0083224, 0.0048962, -0.0084953, 0.0049816, -0.0133039, 0.0133915
5: -0.0054995, 0.0134339, -0.0061471, 0.0129294, -0.0184289, 0.0195810
6: 0.0018522, 0.0115473, 0.0015682, 0.0116180, -0.0097658, 0.0099791
7: -0.0275629, -0.0047229, -0.0259505, -0.0043448, -0.0232180, 0.0212276
8: 0.9448199, 1.0093627, 0.9498057, 1.0105300, -0.0657101, 0.0595570
9: -0.0039751, 0.0160963, -0.0043172, 0.0152547, -0.0192297, 0.0204135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0490010, upper bound: 0.0539010
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0490010, upper bound: 0.0588564
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011749, 0.0030719, -0.0214184, 0.0025814, -0.0036551, 0.0244378
1: -0.0027454, 0.0056294, -0.0139996, 0.0051509, -0.0078963, 0.0195562
2: 0.0088036, 0.0213145, 0.0011083, 0.0219362, -0.0131326, 0.0202062
3: -0.0077557, 0.0047887, -0.0050423, 0.0134676, -0.0212233, 0.0098310
4: -0.0080757, 0.0049460, -0.0079947, 0.0148111, -0.0224690, 0.0129408
5: -0.0059825, 0.0126813, 0.0006820, 0.0324653, -0.0384478, 0.0119993
6: 0.0020863, 0.0114466, -0.0035418, 0.0114003, -0.0093141, 0.0149346
7: -0.0259291, -0.0053234, -0.0361772, -0.0030962, -0.0228329, 0.0307712
8: 0.9495008, 1.0077038, 0.9362923, 1.0045202, -0.0550194, 0.0714115
9: -0.0034871, 0.0152152, -0.0032629, 0.0528350, -0.0554738, 0.0184782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0533491, upper bound: 0.0539762
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0533491, upper bound: 0.0539762
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011446, 0.0029636, -0.0248372, 0.0024876, -0.0035427, 0.0277272
1: -0.0027388, 0.0055067, -0.0158887, 0.0050274, -0.0077662, 0.0212686
2: 0.0088981, 0.0212129, -0.0021206, 0.0221098, -0.0132116, 0.0233336
3: -0.0072268, 0.0047778, -0.0047698, 0.0149774, -0.0222041, 0.0095475
4: -0.0080101, 0.0043526, -0.0093098, 0.0169430, -0.0244823, 0.0136624
5: -0.0047621, 0.0126739, 0.0012302, 0.0357191, -0.0404811, 0.0114437
6: 0.0024315, 0.0114198, -0.0053597, 0.0113611, -0.0089296, 0.0166929
7: -0.0259130, -0.0056405, -0.0378805, 0.0006147, -0.0265277, 0.0320227
8: 0.9495472, 1.0068851, 0.9339926, 1.0036761, -0.0541289, 0.0728925
9: -0.0033574, 0.0149074, -0.0030728, 0.0594416, -0.0619372, 0.0179802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0532913, upper bound: 0.0538005
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0532913, upper bound: 0.0538005
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0008014, 0.0034807, -0.0098264, 0.0031120, -0.0038212, 0.0132543
1: -0.0024311, 0.0061548, -0.0075691, 0.0057608, -0.0081919, 0.0136400
2: 0.0081961, 0.0210610, 0.0084472, 0.0215898, -0.0133937, 0.0126138
3: -0.0091564, 0.0043978, -0.0073095, 0.0083335, -0.0174899, 0.0117073
4: -0.0084970, 0.0060685, -0.0083021, 0.0089504, -0.0173517, 0.0143706
5: -0.0087558, 0.0123280, -0.0047104, 0.0213889, -0.0301447, 0.0170384
6: 0.0010245, 0.0116187, 0.0012187, 0.0115257, -0.0105012, 0.0104000
7: -0.0251621, -0.0039934, -0.0303789, -0.0050218, -0.0201404, 0.0261823
8: 0.9516983, 1.0113472, 0.9441206, 1.0086020, -0.0569037, 0.0672266
9: -0.0043206, 0.0152509, -0.0038702, 0.0310380, -0.0341592, 0.0191211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0514873, upper bound: 0.0476542
time: 1.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0514420, upper bound: 0.0475900
time: 2.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0008618, 0.0035044, -0.0202881, 0.0029580, -0.0037420, 0.0237102
1: -0.0025923, 0.0061823, -0.0133526, 0.0056039, -0.0081961, 0.0191773
2: 0.0081748, 0.0212892, 0.0021951, 0.0220946, -0.0138775, 0.0190941
3: -0.0092483, 0.0045825, -0.0065573, 0.0129553, -0.0222036, 0.0111398
4: -0.0085118, 0.0062484, -0.0082693, 0.0149685, -0.0227621, 0.0145177
5: -0.0089636, 0.0125091, -0.0032561, 0.0313507, -0.0403143, 0.0157653
6: 0.0009438, 0.0116247, -0.0033184, 0.0115023, -0.0105585, 0.0146635
7: -0.0255554, -0.0039237, -0.0355937, -0.0036653, -0.0218900, 0.0309903
8: 0.9505717, 1.0115283, 0.9370800, 1.0075018, -0.0569302, 0.0744482
9: -0.0043499, 0.0155538, -0.0037572, 0.0511200, -0.0538881, 0.0193110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0487430, upper bound: 0.0525627
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0487430, upper bound: 0.0570115
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0008296, 0.0032968, -0.0034815, 0.0037421, -0.0044716, 0.0066957
1: -0.0025635, 0.0059931, -0.0040473, 0.0067352, -0.0092987, 0.0100405
2: 0.0082290, 0.0210260, 0.0069870, 0.0214225, -0.0131935, 0.0140390
3: -0.0079726, 0.0045469, -0.0083342, 0.0055223, -0.0134949, 0.0128811
4: -0.0084742, 0.0046795, -0.0093297, 0.0060128, -0.0144870, 0.0140092
5: -0.0059143, 0.0124768, -0.0055627, 0.0153228, -0.0212371, 0.0180395
6: 0.0016675, 0.0116094, 0.0006722, 0.0119563, -0.0102888, 0.0109371
7: -0.0254852, -0.0044353, -0.0272034, -0.0026548, -0.0228304, 0.0225979
8: 0.9507726, 1.0102998, 0.9484078, 1.0154892, -0.0647166, 0.0618920
9: -0.0042755, 0.0147465, -0.0059560, 0.0192886, -0.0230392, 0.0206932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0491083, upper bound: 0.0468113
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0528051, upper bound: 0.0479876
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0008296, 0.0032968, -0.0225668, 0.0033974, -0.0040890, 0.0257785
1: -0.0025635, 0.0059931, -0.0146066, 0.0063207, -0.0088842, 0.0203823
2: 0.0082290, 0.0210260, 0.0000472, 0.0222601, -0.0140311, 0.0209788
3: -0.0079726, 0.0045469, -0.0069971, 0.0139587, -0.0219313, 0.0115440
4: -0.0084742, 0.0046795, -0.0090796, 0.0165213, -0.0245472, 0.0137591
5: -0.0059143, 0.0124768, -0.0031102, 0.0335108, -0.0394251, 0.0155870
6: 0.0016675, 0.0116094, -0.0046010, 0.0118367, -0.0101692, 0.0160394
7: -0.0254852, -0.0044353, -0.0367245, -0.0012310, -0.0242542, 0.0319061
8: 0.9507726, 1.0102998, 0.9355533, 1.0125802, -0.0618076, 0.0747465
9: -0.0042755, 0.0147465, -0.0053767, 0.0556298, -0.0588316, 0.0201232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0491083, upper bound: 0.0468113
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0528051, upper bound: 0.0479876
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010050, 0.0029286, -0.0010291, 0.0037296, -0.0046538, 0.0038806
1: -0.0033846, 0.0055435, -0.0026767, 0.0067113, -0.0100959, 0.0082202
2: 0.0086365, 0.0218859, 0.0070395, 0.0212549, -0.0126184, 0.0148464
3: -0.0063726, 0.0054716, -0.0083402, 0.0046956, -0.0110681, 0.0138119
4: -0.0081915, 0.0033069, -0.0092993, 0.0046208, -0.0128124, 0.0126062
5: -0.0020846, 0.0133998, -0.0056008, 0.0126041, -0.0146887, 0.0190006
6: 0.0027484, 0.0114939, 0.0007981, 0.0119463, -0.0091978, 0.0106958
7: -0.0274890, -0.0055842, -0.0257615, -0.0027087, -0.0247040, 0.0201773
8: 0.9450317, 1.0072473, 0.9499810, 1.0153438, -0.0703121, 0.0572663
9: -0.0037163, 0.0153011, -0.0059076, 0.0150903, -0.0188065, 0.0210500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0469998, upper bound: 0.0453269
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0515857, upper bound: 0.0478346
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010050, 0.0029286, -0.0008287, 0.0039388, -0.0049174, 0.0036896
1: -0.0033846, 0.0055435, -0.0025592, 0.0069437, -0.0103283, 0.0081027
2: 0.0086365, 0.0218859, 0.0068715, 0.0213402, -0.0127036, 0.0150143
3: -0.0063726, 0.0054716, -0.0094903, 0.0045421, -0.0109147, 0.0149619
4: -0.0081915, 0.0033069, -0.0094158, 0.0059673, -0.0141588, 0.0127227
5: -0.0020846, 0.0133998, -0.0083329, 0.0124720, -0.0145566, 0.0217328
6: 0.0027484, 0.0114939, 0.0001046, 0.0119938, -0.0092454, 0.0113893
7: -0.0274890, -0.0055842, -0.0254748, -0.0020993, -0.0253896, 0.0198906
8: 0.9450317, 1.0072473, 0.9508027, 1.0169210, -0.0718893, 0.0564446
9: -0.0037163, 0.0153011, -0.0061380, 0.0156631, -0.0193794, 0.0212361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0469998, upper bound: 0.0453269
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0515857, upper bound: 0.0478346
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0014686, 0.0033161, -0.0153866, 0.0036016, -0.0049351, 0.0185844
1: -0.0029347, 0.0060212, -0.0093727, 0.0065873, -0.0095220, 0.0151528
2: 0.0082038, 0.0212416, 0.0070570, 0.0273891, -0.0185079, 0.0141846
3: -0.0080429, 0.0047469, -0.0078550, 0.0097768, -0.0178197, 0.0126018
4: -0.0084896, 0.0050842, -0.0093480, 0.0110466, -0.0193817, 0.0144322
5: -0.0060643, 0.0132532, -0.0044104, 0.0261945, -0.0322588, 0.0176637
6: 0.0015789, 0.0116156, -0.0018765, 0.0119197, -0.0103408, 0.0134921
7: -0.0261200, -0.0043699, -0.0320052, -0.0026986, -0.0234215, 0.0269523
8: 0.9496977, 1.0104638, 0.9402258, 1.0144188, -0.0647212, 0.0702380
9: -0.0043059, 0.0155469, -0.0058618, 0.0375356, -0.0405309, 0.0214087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510325, upper bound: 0.0539561
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510325, upper bound: 0.0577429
time: 1.31 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0014686, 0.0033161, -0.0409116, 0.0032400, -0.0045708, 0.0441185
1: -0.0029347, 0.0060212, -0.0199436, 0.0175704, -0.0205050, 0.0256447
2: 0.0082038, 0.0212416, -0.0090715, 0.0440980, -0.0351053, 0.0303131
3: -0.0080429, 0.0047469, -0.0070011, 0.0182233, -0.0262662, 0.0117480
4: -0.0084896, 0.0050842, -0.0292779, 0.0223008, -0.0301353, 0.0343621
5: -0.0060643, 0.0132532, -0.0015212, 0.0491768, -0.0552411, 0.0147745
6: 0.0015789, 0.0116156, -0.0174223, 0.0120991, -0.0105202, 0.0288698
7: -0.0261200, -0.0043699, -0.0415367, 0.0346482, -0.0607683, 0.0364662
8: 0.9496977, 1.0104638, 0.9225824, 1.0114139, -0.0617163, 0.0874598
9: -0.0043059, 0.0155469, -0.0079371, 0.0741754, -0.0772647, 0.0234840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510325, upper bound: 0.0486563
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0510325, upper bound: 0.0577429
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0138208, 0.0028076, -0.0209890, 0.0031336, -0.0167969, 0.0236541
1: -0.0100958, 0.0054048, -0.0117047, 0.0065780, -0.0166738, 0.0171095
2: 0.0087692, 0.0220573, 0.0050179, 0.0309291, -0.0221598, 0.0170394
3: -0.0060506, 0.0058665, -0.0064314, 0.0116367, -0.0176873, 0.0122979
4: -0.0080618, 0.0149522, -0.0136064, 0.0128382, -0.0209001, 0.0285586
5: -0.0013510, 0.0316810, -0.0014561, 0.0312645, -0.0326155, 0.0331371
6: -0.0002549, 0.0114410, -0.0049014, 0.0117135, -0.0119684, 0.0163423
7: -0.0357666, -0.0059631, -0.0341079, 0.0034580, -0.0392246, 0.0281275
8: 0.9435549, 1.0061584, 0.9363336, 1.0101043, -0.0665494, 0.0698248
9: -0.0034597, 0.0315209, -0.0048727, 0.0451955, -0.0486492, 0.0363936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0534410, upper bound: 0.0526704
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0532431, upper bound: 0.0526130
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0139164, 0.0029510, -0.0152867, 0.0035277, -0.0172778, 0.0181033
1: -0.0101451, 0.0055963, -0.0093333, 0.0064841, -0.0166291, 0.0149296
2: 0.0085520, 0.0221228, 0.0071891, 0.0273039, -0.0187519, 0.0149338
3: -0.0065285, 0.0058755, -0.0076594, 0.0097447, -0.0162732, 0.0135349
4: -0.0082081, 0.0152729, -0.0092538, 0.0108855, -0.0190936, 0.0245266
5: -0.0023325, 0.0318078, -0.0040501, 0.0261088, -0.0284413, 0.0358579
6: -0.0004441, 0.0115007, -0.0017558, 0.0118832, -0.0123273, 0.0132565
7: -0.0358330, -0.0054864, -0.0319696, -0.0029582, -0.0328748, 0.0264832
8: 0.9435127, 1.0074497, 0.9402916, 1.0137031, -0.0701904, 0.0671581
9: -0.0037491, 0.0317911, -0.0056812, 0.0373292, -0.0410783, 0.0374724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0494310, upper bound: 0.0531694
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0494310, upper bound: 0.0574525
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0008020, 0.0035308, -0.0018529, 0.0037331, -0.0044455, 0.0053679
1: -0.0024341, 0.0062362, -0.0031492, 0.0067223, -0.0091564, 0.0093854
2: 0.0080695, 0.0210688, 0.0070088, 0.0213236, -0.0132541, 0.0140600
3: -0.0092309, 0.0044012, -0.0082916, 0.0048040, -0.0140349, 0.0126928
4: -0.0085848, 0.0060685, -0.0093176, 0.0051582, -0.0137431, 0.0153862
5: -0.0088306, 0.0123313, -0.0054120, 0.0137757, -0.0226063, 0.0177433
6: 0.0009134, 0.0116545, 0.0007720, 0.0119537, -0.0110403, 0.0108825
7: -0.0251694, -0.0037980, -0.0263936, -0.0026865, -0.0223935, 0.0225956
8: 0.9516774, 1.0119243, 0.9495013, 1.0154150, -0.0637375, 0.0624230
9: -0.0044943, 0.0152599, -0.0059435, 0.0161724, -0.0206072, 0.0212034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0515243, upper bound: 0.0467692
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0512589, upper bound: 0.0467157
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0008020, 0.0035308, -0.0209220, 0.0033891, -0.0040639, 0.0244297
1: -0.0024341, 0.0062362, -0.0137001, 0.0063091, -0.0087432, 0.0198804
2: 0.0080695, 0.0210688, 0.0015987, 0.0221588, -0.0140893, 0.0194700
3: -0.0092309, 0.0044012, -0.0069636, 0.0132336, -0.0224645, 0.0113648
4: -0.0085848, 0.0060685, -0.0090696, 0.0154457, -0.0236554, 0.0151381
5: -0.0088306, 0.0123313, -0.0029581, 0.0319494, -0.0407800, 0.0152895
6: 0.0009134, 0.0116545, -0.0037152, 0.0118342, -0.0109209, 0.0152471
7: -0.0251694, -0.0037980, -0.0359071, -0.0027423, -0.0224271, 0.0321091
8: 0.9516774, 1.0119243, 0.9366570, 1.0125115, -0.0608341, 0.0752673
9: -0.0044943, 0.0152599, -0.0053649, 0.0524129, -0.0559567, 0.0206248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0515243, upper bound: 0.0467692
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0512589, upper bound: 0.0467157
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0009775, 0.0031802, -0.0015546, 0.0035150, -0.0043945, 0.0046897
1: -0.0032558, 0.0058143, -0.0030028, 0.0064748, -0.0097307, 0.0088171
2: 0.0084804, 0.0218942, 0.0071822, 0.0210696, -0.0125892, 0.0147120
3: -0.0075611, 0.0053266, -0.0072385, 0.0047537, -0.0123148, 0.0125651
4: -0.0082998, 0.0046393, -0.0091987, 0.0036546, -0.0119544, 0.0138380
5: -0.0050613, 0.0132551, -0.0028714, 0.0134286, -0.0184898, 0.0161264
6: 0.0019696, 0.0115381, 0.0014477, 0.0119052, -0.0099356, 0.0100905
7: -0.0271747, -0.0048714, -0.0262118, -0.0033266, -0.0237489, 0.0213404
8: 0.9459321, 1.0090493, 0.9496391, 1.0137659, -0.0678338, 0.0594102
9: -0.0039305, 0.0157304, -0.0057087, 0.0150936, -0.0190241, 0.0214391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0469424, upper bound: 0.0474952
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0469424, upper bound: 0.0477142
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0009760, 0.0030750, -0.0050871, 0.0034399, -0.0043244, 0.0080898
1: -0.0032485, 0.0056938, -0.0049617, 0.0063765, -0.0096250, 0.0106555
2: 0.0085732, 0.0218136, 0.0072912, 0.0212397, -0.0126665, 0.0145225
3: -0.0070846, 0.0053183, -0.0070201, 0.0062471, -0.0133318, 0.0123384
4: -0.0082355, 0.0041145, -0.0091183, 0.0055115, -0.0137469, 0.0132328
5: -0.0039585, 0.0132468, -0.0024566, 0.0168977, -0.0208562, 0.0157034
6: 0.0022899, 0.0115119, 0.0015348, 0.0118689, -0.0095790, 0.0099770
7: -0.0271567, -0.0051837, -0.0280279, -0.0035743, -0.0235824, 0.0228442
8: 0.9459836, 1.0082393, 0.9472948, 1.0130681, -0.0670844, 0.0609445
9: -0.0038032, 0.0154828, -0.0055330, 0.0216741, -0.0249949, 0.0210158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0466613, upper bound: 0.0464569
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0466613, upper bound: 0.0465268
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0008691, 0.0035544, -0.0133143, 0.0035928, -0.0043525, 0.0168144
1: -0.0025954, 0.0062635, -0.0085165, 0.0065729, -0.0091683, 0.0146654
2: 0.0080485, 0.0212971, 0.0070768, 0.0260291, -0.0173878, 0.0142202
3: -0.0093220, 0.0045868, -0.0077708, 0.0090921, -0.0184141, 0.0123575
4: -0.0085994, 0.0062467, -0.0093182, 0.0100668, -0.0185958, 0.0155648
5: -0.0090370, 0.0125127, -0.0042641, 0.0243330, -0.0333701, 0.0167768
6: 0.0008334, 0.0116605, -0.0007136, 0.0119170, -0.0110836, 0.0123741
7: -0.0255630, -0.0037287, -0.0312332, -0.0030288, -0.0225343, 0.0271914
8: 0.9505497, 1.0121034, 0.9416549, 1.0143446, -0.0637949, 0.0704485
9: -0.0045231, 0.0155630, -0.0058295, 0.0345138, -0.0377331, 0.0213925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0571108, upper bound: 0.0560705
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0583449, upper bound: 0.0577749
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0008691, 0.0035544, -0.0388449, 0.0032319, -0.0039893, 0.0423520
1: -0.0025954, 0.0062635, -0.0190900, 0.0164155, -0.0190109, 0.0251331
2: 0.0080485, 0.0212971, -0.0076106, 0.0427237, -0.0338281, 0.0289076
3: -0.0093220, 0.0045868, -0.0068480, 0.0175406, -0.0268626, 0.0114347
4: -0.0085994, 0.0062467, -0.0276517, 0.0212702, -0.0292129, 0.0338983
5: -0.0090370, 0.0125127, -0.0013583, 0.0473209, -0.0563579, 0.0138710
6: 0.0008334, 0.0116605, -0.0161044, 0.0119134, -0.0110800, 0.0275067
7: -0.0255630, -0.0037287, -0.0407670, 0.0313943, -0.0569573, 0.0366836
8: 0.9505497, 1.0121034, 0.9240071, 1.0113469, -0.0607972, 0.0880964
9: -0.0045231, 0.0155630, -0.0067655, 0.0711407, -0.0744223, 0.0223284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0571108, upper bound: 0.0560705
time: 2.25 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0583449, upper bound: 0.0577749
time: 1.40 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0026564, 0.0030676, -0.0188306, 0.0031255, -0.0056642, 0.0217983
1: -0.0034092, 0.0056955, -0.0108128, 0.0060066, -0.0094158, 0.0165083
2: 0.0086594, 0.0220727, 0.0065439, 0.0295013, -0.0204466, 0.0155287
3: -0.0072080, 0.0056975, -0.0063575, 0.0109235, -0.0181315, 0.0120551
4: -0.0081757, 0.0044356, -0.0119082, 0.0117768, -0.0199525, 0.0163438
5: -0.0044498, 0.0134275, -0.0013171, 0.0293255, -0.0337753, 0.0147446
6: 0.0022522, 0.0114874, -0.0035472, 0.0117108, -0.0094586, 0.0150347
7: -0.0275490, -0.0052131, -0.0333037, 0.0001073, -0.0276563, 0.0280906
8: 0.9448596, 1.0080385, 0.9378222, 1.0100350, -0.0651755, 0.0702163
9: -0.0036850, 0.0158833, -0.0048457, 0.0420352, -0.0457202, 0.0207290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0531691, upper bound: 0.0526445
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0529617, upper bound: 0.0525849
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0026798, 0.0032166, -0.0132170, 0.0035188, -0.0060758, 0.0163468
1: -0.0034149, 0.0058919, -0.0084782, 0.0064693, -0.0098842, 0.0143702
2: 0.0084479, 0.0221428, 0.0072087, 0.0259448, -0.0174061, 0.0149342
3: -0.0077144, 0.0057066, -0.0075777, 0.0090609, -0.0167753, 0.0132843
4: -0.0083224, 0.0048962, -0.0092246, 0.0099054, -0.0182277, 0.0141208
5: -0.0054995, 0.0134339, -0.0039067, 0.0242498, -0.0297494, 0.0173406
6: 0.0018522, 0.0115473, -0.0005725, 0.0118804, -0.0100282, 0.0121198
7: -0.0275629, -0.0047229, -0.0311987, -0.0032846, -0.0242783, 0.0264758
8: 0.9448199, 1.0093627, 0.9417187, 1.0136282, -0.0688084, 0.0676440
9: -0.0039751, 0.0160963, -0.0056494, 0.0343098, -0.0382848, 0.0217457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0482200, upper bound: 0.0527781
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0482200, upper bound: 0.0483306
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010690, 0.0038966, -0.0010234, 0.0032413, -0.0042205, 0.0048483
1: -0.0026827, 0.0069037, -0.0026833, 0.0059199, -0.0086026, 0.0095870
2: 0.0068935, 0.0214145, 0.0083235, 0.0211573, -0.0142638, 0.0130909
3: -0.0091286, 0.0047069, -0.0077717, 0.0047021, -0.0138307, 0.0124786
4: -0.0094005, 0.0055321, -0.0084087, 0.0045508, -0.0139513, 0.0139408
5: -0.0073826, 0.0126108, -0.0054948, 0.0126114, -0.0199940, 0.0181056
6: 0.0002869, 0.0119876, 0.0018151, 0.0115826, -0.0112956, 0.0101725
7: -0.0257761, -0.0022118, -0.0257774, -0.0046222, -0.0209839, 0.0234483
8: 0.9499394, 1.0166347, 0.9499354, 1.0097767, -0.0598373, 0.0666993
9: -0.0061078, 0.0155770, -0.0041458, 0.0148432, -0.0208420, 0.0197228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0504925, upper bound: 0.0535316
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0504925, upper bound: 0.0535316
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0115032, 0.0037533, -0.0007828, 0.0034666, -0.0149082, 0.0044452
1: -0.0077566, 0.0067545, -0.0023444, 0.0061701, -0.0138552, 0.0090989
2: 0.0069502, 0.0249545, 0.0081278, 0.0209114, -0.0139613, 0.0165115
3: -0.0085695, 0.0084877, -0.0089043, 0.0043003, -0.0128698, 0.0173920
4: -0.0093989, 0.0099085, -0.0085444, 0.0057257, -0.0151246, 0.0184529
5: -0.0061532, 0.0226809, -0.0080932, 0.0122306, -0.0183837, 0.0307740
6: -0.0003981, 0.0119517, 0.0011283, 0.0116380, -0.0120361, 0.0108234
7: -0.0305480, -0.0025912, -0.0249506, -0.0039691, -0.0262463, 0.0223594
8: 0.9429232, 1.0155566, 0.9523044, 1.0114783, -0.0685551, 0.0632522
9: -0.0059918, 0.0322244, -0.0044144, 0.0149674, -0.0209592, 0.0354766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0540103, upper bound: 0.0510472
time: 1.24 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0540103, upper bound: 0.0510472
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0115032, 0.0037533, -0.0025821, 0.0033348, -0.0147213, 0.0061935
1: -0.0077566, 0.0067545, -0.0035581, 0.0060450, -0.0136705, 0.0103126
2: 0.0069502, 0.0249545, 0.0081753, 0.0213194, -0.0143692, 0.0167677
3: -0.0085695, 0.0084877, -0.0081291, 0.0051295, -0.0136990, 0.0166168
4: -0.0093989, 0.0099085, -0.0085057, 0.0057372, -0.0151361, 0.0184142
5: -0.0061532, 0.0226809, -0.0062653, 0.0144801, -0.0206332, 0.0289462
6: -0.0003981, 0.0119517, 0.0014489, 0.0116207, -0.0120188, 0.0105028
7: -0.0305480, -0.0025912, -0.0267623, -0.0043102, -0.0257701, 0.0241711
8: 0.9429232, 1.0155566, 0.9490036, 1.0106113, -0.0676881, 0.0665530
9: -0.0059918, 0.0322244, -0.0043304, 0.0174938, -0.0234856, 0.0353704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0540103, upper bound: 0.0580074
time: 1.33 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0540103, upper bound: 0.0580074
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010055, 0.0037225, -0.0174452, 0.0028583, -0.0037773, 0.0210381
1: -0.0026674, 0.0067041, -0.0117968, 0.0055004, -0.0081678, 0.0181765
2: 0.0070406, 0.0212280, 0.0048672, 0.0218083, -0.0147677, 0.0163608
3: -0.0082931, 0.0046825, -0.0059947, 0.0117086, -0.0200017, 0.0106772
4: -0.0092985, 0.0045450, -0.0082386, 0.0126691, -0.0214340, 0.0127836
5: -0.0054776, 0.0125936, -0.0015148, 0.0286710, -0.0341485, 0.0141085
6: 0.0008246, 0.0119459, -0.0015983, 0.0114970, -0.0106724, 0.0134047
7: -0.0257388, -0.0027275, -0.0341909, -0.0054492, -0.0202896, 0.0307851
8: 0.9500462, 1.0152988, 0.9389739, 1.0068756, -0.0568295, 0.0763249
9: -0.0059060, 0.0150310, -0.0037310, 0.0453238, -0.0500063, 0.0187620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0454287, upper bound: 0.0481373
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0454287, upper bound: 0.0518683
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0112739, 0.0035868, -0.0116088, 0.0031206, -0.0142762, 0.0150586
1: -0.0076721, 0.0065591, -0.0085524, 0.0057737, -0.0134458, 0.0151115
2: 0.0071062, 0.0246985, 0.0084228, 0.0216977, -0.0145915, 0.0162757
3: -0.0077667, 0.0084172, -0.0073520, 0.0091198, -0.0168865, 0.0157692
4: -0.0092824, 0.0091730, -0.0083152, 0.0100457, -0.0193280, 0.0174882
5: -0.0043286, 0.0224972, -0.0048696, 0.0230826, -0.0274113, 0.0273668
6: 0.0001713, 0.0119113, 0.0006557, 0.0115281, -0.0113568, 0.0112556
7: -0.0304718, -0.0031031, -0.0312656, -0.0049782, -0.0254936, 0.0281625
8: 0.9430643, 1.0142572, 0.9429235, 1.0086730, -0.0656087, 0.0713336
9: -0.0057839, 0.0315646, -0.0038818, 0.0345003, -0.0402842, 0.0349934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0484318, upper bound: 0.0449764
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0525617, upper bound: 0.0477636
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0112739, 0.0035868, -0.0219949, 0.0029667, -0.0141070, 0.0254190
1: -0.0076721, 0.0065591, -0.0142934, 0.0056166, -0.0132887, 0.0208525
2: 0.0071062, 0.0246985, 0.0005850, 0.0222009, -0.0150947, 0.0241135
3: -0.0077667, 0.0084172, -0.0065984, 0.0137078, -0.0214745, 0.0150156
4: -0.0092824, 0.0091730, -0.0082808, 0.0160826, -0.0253649, 0.0174538
5: -0.0043286, 0.0224972, -0.0034207, 0.0329712, -0.0372998, 0.0259179
6: 0.0001713, 0.0119113, -0.0042360, 0.0115049, -0.0113336, 0.0161473
7: -0.0304718, -0.0031031, -0.0364420, -0.0019541, -0.0285177, 0.0333390
8: 0.9430643, 1.0142572, 0.9359347, 1.0075740, -0.0645097, 0.0783225
9: -0.0057839, 0.0315646, -0.0037695, 0.0544553, -0.0601256, 0.0353340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0484318, upper bound: 0.0535126
time: 2.54 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0525617, upper bound: 0.0560632
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0058113, 0.0037854, -0.0007669, 0.0034569, -0.0092204, 0.0044897
1: -0.0053268, 0.0067786, -0.0022699, 0.0061588, -0.0114856, 0.0090485
2: 0.0069640, 0.0216234, 0.0081368, 0.0208096, -0.0138456, 0.0134866
3: -0.0086637, 0.0065466, -0.0088729, 0.0042163, -0.0128801, 0.0154195
4: -0.0093407, 0.0076030, -0.0085381, 0.0056638, -0.0150045, 0.0161412
5: -0.0065306, 0.0175266, -0.0080308, 0.0121468, -0.0186774, 0.0255575
6: 0.0003315, 0.0119571, 0.0011593, 0.0116355, -0.0113039, 0.0107978
7: -0.0283571, -0.0025355, -0.0247688, -0.0039984, -0.0242207, 0.0222333
8: 0.9468502, 1.0157501, 0.9528254, 1.0114015, -0.0645513, 0.0629247
9: -0.0059601, 0.0239171, -0.0044020, 0.0148420, -0.0208021, 0.0274559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0575010, upper bound: 0.0520676
time: 1.45 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0597646, upper bound: 0.0538189
time: 1.36 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0104882, 0.0038080, -0.0012478, 0.0033248, -0.0137399, 0.0049550
1: -0.0073317, 0.0068137, -0.0028088, 0.0060305, -0.0133622, 0.0096225
2: 0.0069121, 0.0243535, 0.0081971, 0.0212170, -0.0143049, 0.0161468
3: -0.0088798, 0.0081496, -0.0080795, 0.0047268, -0.0136066, 0.0162290
4: -0.0094169, 0.0098057, -0.0084953, 0.0049816, -0.0143984, 0.0183009
5: -0.0069544, 0.0217571, -0.0061471, 0.0129294, -0.0198837, 0.0279042
6: -0.0003667, 0.0119633, 0.0015682, 0.0116180, -0.0119847, 0.0103951
7: -0.0301648, -0.0024417, -0.0259505, -0.0043448, -0.0256523, 0.0235088
8: 0.9436325, 1.0159341, 0.9498057, 1.0105300, -0.0668975, 0.0661284
9: -0.0060386, 0.0309371, -0.0043172, 0.0152547, -0.0212933, 0.0344298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0562547, upper bound: 0.0575275
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0562547, upper bound: 0.0611081
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0312944, 0.0034468, -0.0007669, 0.0034569, -0.0347020, 0.0041184
1: -0.0159546, 0.0123164, -0.0022699, 0.0061588, -0.0219524, 0.0145863
2: -0.0022634, 0.0378050, 0.0081368, 0.0208096, -0.0230730, 0.0289875
3: -0.0078546, 0.0150377, -0.0088729, 0.0042163, -0.0120709, 0.0239106
4: -0.0217182, 0.0183330, -0.0085381, 0.0056638, -0.0273820, 0.0264328
5: -0.0040647, 0.0405043, -0.0080308, 0.0121468, -0.0162115, 0.0485351
6: -0.0115726, 0.0118427, 0.0011593, 0.0116355, -0.0229555, 0.0106834
7: -0.0379400, 0.0197362, -0.0247688, -0.0039984, -0.0337085, 0.0445049
8: 0.9292403, 1.0129187, 0.9528254, 1.0114015, -0.0821612, 0.0600933
9: -0.0056061, 0.0605253, -0.0044020, 0.0148420, -0.0204482, 0.0639226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0485324, upper bound: 0.0465314
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0530843, upper bound: 0.0501616
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0360842, 0.0034687, -0.0012478, 0.0033248, -0.0393365, 0.0045805
1: -0.0179331, 0.0149810, -0.0028088, 0.0060305, -0.0238407, 0.0177899
2: -0.0056495, 0.0409748, 0.0081971, 0.0212170, -0.0268665, 0.0322574
3: -0.0081812, 0.0166201, -0.0080795, 0.0047268, -0.0129079, 0.0246996
4: -0.0254870, 0.0207153, -0.0084953, 0.0049816, -0.0304686, 0.0288330
5: -0.0045041, 0.0448058, -0.0061471, 0.0129294, -0.0174335, 0.0509529
6: -0.0146179, 0.0118493, 0.0015682, 0.0116180, -0.0261098, 0.0102811
7: -0.0397239, 0.0272521, -0.0259505, -0.0043448, -0.0351606, 0.0532026
8: 0.9259380, 1.0131009, 0.9498057, 1.0105300, -0.0845920, 0.0632952
9: -0.0056790, 0.0675555, -0.0043172, 0.0152547, -0.0209337, 0.0710734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0486342, upper bound: 0.0531497
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0486342, upper bound: 0.0585239
time: 1.31 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0008271, 0.0039319, -0.0157553, 0.0028500, -0.0036011, 0.0196139
1: -0.0025517, 0.0069372, -0.0108652, 0.0054887, -0.0080404, 0.0175764
2: 0.0068727, 0.0213160, 0.0064614, 0.0217006, -0.0148280, 0.0148546
3: -0.0094412, 0.0045337, -0.0059610, 0.0109636, -0.0204048, 0.0104947
4: -0.0094150, 0.0058925, -0.0082282, 0.0115582, -0.0204494, 0.0141207
5: -0.0082095, 0.0124636, -0.0013749, 0.0270663, -0.0352758, 0.0138385
6: 0.0001311, 0.0119935, -0.0006895, 0.0114945, -0.0113634, 0.0125712
7: -0.0254565, -0.0021171, -0.0333509, -0.0056763, -0.0197802, 0.0308576
8: 0.9508550, 1.0168785, 0.9401081, 1.0068071, -0.0559521, 0.0767704
9: -0.0061365, 0.0156062, -0.0037191, 0.0420164, -0.0469054, 0.0193253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0470584, upper bound: 0.0503109
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0457966, upper bound: 0.0495824
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0081694, 0.0038125, -0.0098264, 0.0031120, -0.0111759, 0.0135650
1: -0.0063727, 0.0068106, -0.0075691, 0.0057608, -0.0121335, 0.0143797
2: 0.0069365, 0.0228600, 0.0084472, 0.0215898, -0.0146533, 0.0144128
3: -0.0088896, 0.0073829, -0.0073095, 0.0083335, -0.0172231, 0.0146923
4: -0.0093771, 0.0088262, -0.0083021, 0.0089504, -0.0183275, 0.0171284
5: -0.0070434, 0.0196721, -0.0047104, 0.0213889, -0.0284323, 0.0243825
6: -0.0000295, 0.0119601, 0.0012187, 0.0115257, -0.0115551, 0.0107414
7: -0.0293001, -0.0024522, -0.0303789, -0.0050218, -0.0242784, 0.0279268
8: 0.9452331, 1.0159364, 0.9441206, 1.0086020, -0.0633689, 0.0718158
9: -0.0059963, 0.0276116, -0.0038702, 0.0310380, -0.0368394, 0.0311681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0469319, upper bound: 0.0440349
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0522732, upper bound: 0.0477340
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0081694, 0.0038125, -0.0202881, 0.0029580, -0.0110044, 0.0239963
1: -0.0063727, 0.0068106, -0.0133526, 0.0056039, -0.0119765, 0.0201632
2: 0.0069365, 0.0228600, 0.0021951, 0.0220946, -0.0151581, 0.0206649
3: -0.0088896, 0.0073829, -0.0065573, 0.0129553, -0.0218449, 0.0139402
4: -0.0093771, 0.0088262, -0.0082693, 0.0149685, -0.0243456, 0.0170955
5: -0.0070434, 0.0196721, -0.0032561, 0.0313507, -0.0383941, 0.0229283
6: -0.0000295, 0.0119601, -0.0033184, 0.0115023, -0.0115318, 0.0152785
7: -0.0293001, -0.0024522, -0.0355937, -0.0036653, -0.0256348, 0.0331416
8: 0.9452331, 1.0159364, 0.9370800, 1.0075018, -0.0622687, 0.0788563
9: -0.0059963, 0.0276116, -0.0037572, 0.0511200, -0.0565688, 0.0313688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0469319, upper bound: 0.0531018
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0522732, upper bound: 0.0560470
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0150533, 0.0032585, -0.0010435, 0.0037543, -0.0185597, 0.0040415
1: -0.0092427, 0.0061013, -0.0026776, 0.0067177, -0.0150993, 0.0087789
2: 0.0076929, 0.0270878, 0.0070999, 0.0213254, -0.0136326, 0.0182952
3: -0.0069162, 0.0096705, -0.0086233, 0.0046983, -0.0116145, 0.0181879
4: -0.0089405, 0.0104396, -0.0092574, 0.0050641, -0.0140046, 0.0188222
5: -0.0026907, 0.0259120, -0.0063660, 0.0126051, -0.0152958, 0.0322779
6: -0.0014322, 0.0117425, 0.0006735, 0.0119292, -0.0131754, 0.0110690
7: -0.0318880, -0.0039201, -0.0257636, -0.0026794, -0.0270787, 0.0218436
8: 0.9404429, 1.0110722, 0.9499748, 1.0153605, -0.0724269, 0.0610973
9: -0.0049913, 0.0367948, -0.0058247, 0.0153147, -0.0203060, 0.0400930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0531476, upper bound: 0.0497296
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0578016, upper bound: 0.0527901
time: 1.25 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0150533, 0.0032585, -0.0008289, 0.0039715, -0.0188299, 0.0038362
1: -0.0092427, 0.0061013, -0.0025603, 0.0069507, -0.0154063, 0.0086616
2: 0.0076929, 0.0270878, 0.0069348, 0.0214141, -0.0137212, 0.0184787
3: -0.0069162, 0.0096705, -0.0098098, 0.0045433, -0.0114596, 0.0194803
4: -0.0089405, 0.0104396, -0.0093719, 0.0064359, -0.0153764, 0.0189114
5: -0.0026907, 0.0259120, -0.0091453, 0.0124732, -0.0151639, 0.0350573
6: -0.0014322, 0.0117425, -0.0000307, 0.0119759, -0.0131413, 0.0117732
7: -0.0318880, -0.0039201, -0.0254774, -0.0020677, -0.0279601, 0.0215573
8: 0.9404429, 1.0110722, 0.9507951, 1.0169424, -0.0747707, 0.0602770
9: -0.0049913, 0.0367948, -0.0060512, 0.0158998, -0.0208911, 0.0402684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0531476, upper bound: 0.0497296
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0578016, upper bound: 0.0527901
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0091436, 0.0036580, -0.0080579, 0.0034485, -0.0122978, 0.0114551
1: -0.0067847, 0.0066263, -0.0065806, 0.0063170, -0.0124746, 0.0131698
2: 0.0071044, 0.0233770, 0.0075333, 0.0216122, -0.0145078, 0.0158437
3: -0.0081619, 0.0077097, -0.0076539, 0.0075460, -0.0157078, 0.0153207
4: -0.0092689, 0.0086054, -0.0089430, 0.0082386, -0.0175076, 0.0175484
5: -0.0053312, 0.0205680, -0.0046047, 0.0196863, -0.0250174, 0.0251727
6: 0.0003671, 0.0119126, 0.0010017, 0.0117923, -0.0114251, 0.0109110
7: -0.0296717, -0.0029269, -0.0294877, -0.0036863, -0.0255109, 0.0265608
8: 0.9445453, 1.0147034, 0.9453239, 1.0125542, -0.0664699, 0.0686419
9: -0.0057748, 0.0286851, -0.0051615, 0.0279101, -0.0322277, 0.0315617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0524593, upper bound: 0.0500621
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0567813, upper bound: 0.0516498
time: 1.25 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0091436, 0.0036580, -0.0035738, 0.0038372, -0.0127233, 0.0069516
1: -0.0067847, 0.0066263, -0.0040866, 0.0068261, -0.0129597, 0.0107129
2: 0.0071044, 0.0233770, 0.0069737, 0.0215467, -0.0144423, 0.0159432
3: -0.0081619, 0.0077097, -0.0089387, 0.0055562, -0.0137180, 0.0166484
4: -0.0092689, 0.0086054, -0.0093378, 0.0067319, -0.0160009, 0.0178319
5: -0.0053312, 0.0205680, -0.0070296, 0.0153905, -0.0207216, 0.0275976
6: 0.0003671, 0.0119126, 0.0003143, 0.0119590, -0.0115919, 0.0115984
7: -0.0296717, -0.0029269, -0.0272388, -0.0024099, -0.0254394, 0.0243120
8: 0.9445453, 1.0147034, 0.9483601, 1.0160756, -0.0702390, 0.0663432
9: -0.0057748, 0.0286851, -0.0059694, 0.0197743, -0.0245077, 0.0322469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0524593, upper bound: 0.0540369
time: 1.21 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0567813, upper bound: 0.0551687
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010690, 0.0038966, -0.0153866, 0.0036016, -0.0044157, 0.0190569
1: -0.0026827, 0.0069037, -0.0093727, 0.0065873, -0.0092700, 0.0155067
2: 0.0068935, 0.0214145, 0.0070570, 0.0273891, -0.0189063, 0.0143574
3: -0.0091286, 0.0047069, -0.0078550, 0.0097768, -0.0189054, 0.0125619
4: -0.0094005, 0.0055321, -0.0093480, 0.0110466, -0.0197027, 0.0148801
5: -0.0073826, 0.0126108, -0.0044104, 0.0261945, -0.0335771, 0.0170212
6: 0.0002869, 0.0119876, -0.0018765, 0.0119197, -0.0116328, 0.0138352
7: -0.0257761, -0.0022118, -0.0320052, -0.0026986, -0.0230775, 0.0278460
8: 0.9499394, 1.0166347, 0.9402258, 1.0144188, -0.0644795, 0.0750933
9: -0.0061078, 0.0155770, -0.0058618, 0.0375356, -0.0412064, 0.0214388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0500831, upper bound: 0.0524888
time: 1.30 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0500831, upper bound: 0.0524888
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010291, 0.0037296, -0.0409116, 0.0032400, -0.0040360, 0.0443969
1: -0.0026767, 0.0067113, -0.0199436, 0.0175704, -0.0202471, 0.0257742
2: 0.0070395, 0.0212549, -0.0090715, 0.0440980, -0.0353887, 0.0303264
3: -0.0083402, 0.0046956, -0.0070011, 0.0182233, -0.0264541, 0.0116967
4: -0.0092993, 0.0046208, -0.0292779, 0.0223008, -0.0303701, 0.0338987
5: -0.0056008, 0.0126041, -0.0015212, 0.0491768, -0.0547776, 0.0141253
6: 0.0007981, 0.0119463, -0.0174223, 0.0120991, -0.0113010, 0.0289414
7: -0.0257615, -0.0027087, -0.0415367, 0.0346482, -0.0604098, 0.0368001
8: 0.9499810, 1.0153438, 0.9225824, 1.0114139, -0.0614329, 0.0887919
9: -0.0059076, 0.0150903, -0.0079371, 0.0741754, -0.0777650, 0.0230274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0451840, upper bound: 0.0481373
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0494040, upper bound: 0.0518726
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0115032, 0.0037533, -0.0153866, 0.0036016, -0.0148015, 0.0188641
1: -0.0077566, 0.0067545, -0.0093727, 0.0065873, -0.0143438, 0.0161272
2: 0.0069502, 0.0249545, 0.0070570, 0.0273891, -0.0204389, 0.0178975
3: -0.0085695, 0.0084877, -0.0078550, 0.0097768, -0.0183463, 0.0163427
4: -0.0093989, 0.0099085, -0.0093480, 0.0110466, -0.0204455, 0.0192565
5: -0.0061532, 0.0226809, -0.0044104, 0.0261945, -0.0323477, 0.0270913
6: -0.0003981, 0.0119517, -0.0018765, 0.0119197, -0.0123179, 0.0138282
7: -0.0305480, -0.0025912, -0.0320052, -0.0026986, -0.0278494, 0.0294140
8: 0.9429232, 1.0155566, 0.9402258, 1.0144188, -0.0697365, 0.0731457
9: -0.0059918, 0.0322244, -0.0058618, 0.0375356, -0.0421557, 0.0375396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0556762, upper bound: 0.0555834
time: 1.31 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0583184, upper bound: 0.0581363
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0112739, 0.0035868, -0.0409116, 0.0032400, -0.0142327, 0.0442097
1: -0.0076721, 0.0065591, -0.0199436, 0.0175704, -0.0252425, 0.0265027
2: 0.0071062, 0.0246985, -0.0090715, 0.0440980, -0.0369918, 0.0337700
3: -0.0077667, 0.0084172, -0.0070011, 0.0182233, -0.0259516, 0.0154183
4: -0.0092824, 0.0091730, -0.0292779, 0.0223008, -0.0315832, 0.0384509
5: -0.0043286, 0.0224972, -0.0015212, 0.0491768, -0.0535055, 0.0240184
6: 0.0001713, 0.0119113, -0.0174223, 0.0120991, -0.0119278, 0.0292827
7: -0.0304718, -0.0031031, -0.0415367, 0.0346482, -0.0651200, 0.0384337
8: 0.9430643, 1.0142572, 0.9225824, 1.0114139, -0.0673019, 0.0868239
9: -0.0057839, 0.0315646, -0.0079371, 0.0741754, -0.0786752, 0.0395017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0554189, upper bound: 0.0543604
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0580771, upper bound: 0.0567444
time: 1.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0058113, 0.0037854, -0.0018529, 0.0037331, -0.0092939, 0.0054173
1: -0.0053268, 0.0067786, -0.0031492, 0.0067223, -0.0117547, 0.0099278
2: 0.0069640, 0.0216234, 0.0070088, 0.0213236, -0.0143596, 0.0143397
3: -0.0086637, 0.0065466, -0.0082916, 0.0048040, -0.0134678, 0.0148382
4: -0.0093407, 0.0076030, -0.0093176, 0.0051582, -0.0144990, 0.0169207
5: -0.0065306, 0.0175266, -0.0054120, 0.0137757, -0.0203063, 0.0229386
6: 0.0003315, 0.0119571, 0.0007720, 0.0119537, -0.0116222, 0.0111851
7: -0.0283571, -0.0025355, -0.0263936, -0.0026865, -0.0237744, 0.0235870
8: 0.9468502, 1.0157501, 0.9495013, 1.0154150, -0.0685647, 0.0662488
9: -0.0059601, 0.0239171, -0.0059435, 0.0161724, -0.0211067, 0.0277788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0483161, upper bound: 0.0460313
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0537798, upper bound: 0.0477972
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0058113, 0.0037854, -0.0209220, 0.0033891, -0.0089084, 0.0244845
1: -0.0053268, 0.0067786, -0.0137001, 0.0063091, -0.0116359, 0.0198809
2: 0.0069640, 0.0216234, 0.0015987, 0.0221588, -0.0151948, 0.0200247
3: -0.0086637, 0.0065466, -0.0069636, 0.0132336, -0.0218974, 0.0135101
4: -0.0093407, 0.0076030, -0.0090696, 0.0154457, -0.0243757, 0.0166726
5: -0.0065306, 0.0175266, -0.0029581, 0.0319494, -0.0384800, 0.0204847
6: 0.0003315, 0.0119571, -0.0037152, 0.0118342, -0.0115027, 0.0156098
7: -0.0283571, -0.0025355, -0.0359071, -0.0027423, -0.0256148, 0.0328296
8: 0.9468502, 1.0157501, 0.9366570, 1.0125115, -0.0656613, 0.0783794
9: -0.0059601, 0.0239171, -0.0053649, 0.0524129, -0.0564635, 0.0279849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0483161, upper bound: 0.0460312
time: 1.28 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0537798, upper bound: 0.0477972
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0249993, 0.0034340, -0.0015546, 0.0035150, -0.0282396, 0.0047678
1: -0.0159440, 0.0063616, -0.0030028, 0.0064748, -0.0215220, 0.0093644
2: -0.0022444, 0.0224535, 0.0071822, 0.0210696, -0.0233140, 0.0149595
3: -0.0072174, 0.0150291, -0.0072385, 0.0047537, -0.0119711, 0.0220297
4: -0.0094096, 0.0182745, -0.0091987, 0.0036546, -0.0130641, 0.0264047
5: -0.0038945, 0.0358143, -0.0028714, 0.0134286, -0.0173231, 0.0386857
6: -0.0059814, 0.0118402, 0.0014477, 0.0119052, -0.0175316, 0.0103926
7: -0.0379304, 0.0014360, -0.0262118, -0.0033266, -0.0324645, 0.0276479
8: 0.9339253, 1.0128248, 0.9496391, 1.0137659, -0.0777322, 0.0631856
9: -0.0053939, 0.0604605, -0.0057087, 0.0150936, -0.0204874, 0.0639536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0463466, upper bound: 0.0470902
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0463466, upper bound: 0.0475455
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0248074, 0.0033358, -0.0050871, 0.0034399, -0.0279791, 0.0081858
1: -0.0158455, 0.0062441, -0.0049617, 0.0063765, -0.0214329, 0.0112058
2: -0.0020696, 0.0223676, 0.0072912, 0.0212397, -0.0233093, 0.0150764
3: -0.0067936, 0.0149487, -0.0070201, 0.0062471, -0.0130407, 0.0217299
4: -0.0093262, 0.0178593, -0.0091183, 0.0055115, -0.0148377, 0.0263643
5: -0.0027769, 0.0356447, -0.0024566, 0.0168977, -0.0196746, 0.0381013
6: -0.0057517, 0.0118145, 0.0015348, 0.0118689, -0.0174205, 0.0102797
7: -0.0378416, 0.0010779, -0.0280279, -0.0035743, -0.0330767, 0.0291058
8: 0.9340452, 1.0120533, 0.9472948, 1.0130681, -0.0769917, 0.0647585
9: -0.0052694, 0.0599326, -0.0055330, 0.0216741, -0.0263670, 0.0634903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0462078, upper bound: 0.0462078
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0462078, upper bound: 0.0463174
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0104882, 0.0038080, -0.0133143, 0.0035928, -0.0138172, 0.0168761
1: -0.0073317, 0.0068137, -0.0085165, 0.0065729, -0.0139046, 0.0153302
2: 0.0069121, 0.0243535, 0.0070768, 0.0260291, -0.0191170, 0.0172767
3: -0.0088798, 0.0081496, -0.0077708, 0.0090921, -0.0179719, 0.0159203
4: -0.0094169, 0.0098057, -0.0093182, 0.0100668, -0.0194837, 0.0191238
5: -0.0069544, 0.0217571, -0.0042641, 0.0243330, -0.0312874, 0.0260212
6: -0.0003667, 0.0119633, -0.0007136, 0.0119170, -0.0122837, 0.0126769
7: -0.0301648, -0.0024417, -0.0312332, -0.0030288, -0.0271361, 0.0287915
8: 0.9436325, 1.0159341, 0.9416549, 1.0143446, -0.0706809, 0.0729216
9: -0.0060386, 0.0309371, -0.0058295, 0.0345138, -0.0391081, 0.0362308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511345, upper bound: 0.0532298
time: 1.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511345, upper bound: 0.0573219
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0104882, 0.0038080, -0.0388449, 0.0032319, -0.0134556, 0.0424117
1: -0.0073317, 0.0068137, -0.0190900, 0.0164155, -0.0237472, 0.0258696
2: 0.0069121, 0.0243535, -0.0076106, 0.0427237, -0.0358116, 0.0319641
3: -0.0088798, 0.0081496, -0.0068480, 0.0175406, -0.0264204, 0.0149975
4: -0.0094169, 0.0098057, -0.0276517, 0.0212702, -0.0306871, 0.0374573
5: -0.0069544, 0.0217571, -0.0013583, 0.0473209, -0.0542752, 0.0231154
6: -0.0003667, 0.0119633, -0.0161044, 0.0119134, -0.0122801, 0.0280303
7: -0.0301648, -0.0024417, -0.0407670, 0.0313943, -0.0615591, 0.0383253
8: 0.9436325, 1.0159341, 0.9240071, 1.0113469, -0.0677145, 0.0879975
9: -0.0060386, 0.0309371, -0.0067655, 0.0711407, -0.0758356, 0.0377026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511345, upper bound: 0.0532298
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0511345, upper bound: 0.0484598
time: 1.30 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0359013, 0.0033260, -0.0188306, 0.0031255, -0.0387527, 0.0218884
1: -0.0178624, 0.0148457, -0.0108128, 0.0060066, -0.0238690, 0.0256585
2: -0.0055230, 0.0408251, 0.0065439, 0.0295013, -0.0350242, 0.0342811
3: -0.0076364, 0.0165622, -0.0063575, 0.0109235, -0.0185599, 0.0229197
4: -0.0253409, 0.0203691, -0.0119082, 0.0117768, -0.0371177, 0.0322773
5: -0.0034339, 0.0446521, -0.0013171, 0.0293255, -0.0327593, 0.0459692
6: -0.0144183, 0.0117914, -0.0035472, 0.0117108, -0.0261291, 0.0153386
7: -0.0396602, 0.0269011, -0.0333037, 0.0001073, -0.0397675, 0.0602048
8: 0.9260559, 1.0118285, 0.9378222, 1.0100350, -0.0806551, 0.0716919
9: -0.0053721, 0.0671483, -0.0048457, 0.0420352, -0.0474073, 0.0719940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0535281, upper bound: 0.0523795
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0532344, upper bound: 0.0523395
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0360842, 0.0034687, -0.0132170, 0.0035188, -0.0393226, 0.0164329
1: -0.0179331, 0.0149810, -0.0084782, 0.0064693, -0.0244024, 0.0234593
2: -0.0056495, 0.0409748, 0.0072087, 0.0259448, -0.0315943, 0.0337662
3: -0.0081812, 0.0166201, -0.0075777, 0.0090609, -0.0172420, 0.0241979
4: -0.0254870, 0.0207153, -0.0092246, 0.0099054, -0.0353924, 0.0299399
5: -0.0045041, 0.0448058, -0.0039067, 0.0242498, -0.0287539, 0.0487125
6: -0.0146179, 0.0118493, -0.0005725, 0.0118804, -0.0264983, 0.0124218
7: -0.0397239, 0.0272521, -0.0311987, -0.0032846, -0.0364393, 0.0584508
8: 0.9259380, 1.0131009, 0.9417187, 1.0136282, -0.0841662, 0.0708366
9: -0.0056790, 0.0675555, -0.0056494, 0.0343098, -0.0399887, 0.0725611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0481364, upper bound: 0.0518787
time: 1.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0481364, upper bound: 0.0567152
time: 1.02 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.19 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0563027, upper bound: 0.0522557
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0578783, upper bound: 0.0539938
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0550480, upper bound: 0.0577187
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0550480, upper bound: 0.0613858
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0501423, upper bound: 0.0488724
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0530275, upper bound: 0.0506864
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0506780, upper bound: 0.0542632
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0506780, upper bound: 0.0589097
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0487058, upper bound: 0.0515020
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0484687, upper bound: 0.0512766
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0492170, upper bound: 0.0467462
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0523427, upper bound: 0.0482045
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0492170, upper bound: 0.0539965
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0523427, upper bound: 0.0563308
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0559736, upper bound: 0.0522528
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0575259, upper bound: 0.0539796
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0603857, upper bound: 0.0600598
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0612295, upper bound: 0.0612295
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0490711, upper bound: 0.0482550
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0527025, upper bound: 0.0504740
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0490010, upper bound: 0.0539010
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0490010, upper bound: 0.0588564
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0533491, upper bound: 0.0539762
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0533491, upper bound: 0.0539762
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0532913, upper bound: 0.0538005
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0532913, upper bound: 0.0538005
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0514873, upper bound: 0.0476542
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0514420, upper bound: 0.0475900
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0487430, upper bound: 0.0525627
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0487430, upper bound: 0.0570115
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0491083, upper bound: 0.0468113
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0528051, upper bound: 0.0479876
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0491083, upper bound: 0.0468113
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0528051, upper bound: 0.0479876
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0469998, upper bound: 0.0453269
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0515857, upper bound: 0.0478346
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0469998, upper bound: 0.0453269
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0515857, upper bound: 0.0478346
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0510325, upper bound: 0.0539561
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0510325, upper bound: 0.0577429
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0510325, upper bound: 0.0486563
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0510325, upper bound: 0.0577429
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0534410, upper bound: 0.0526704
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0532431, upper bound: 0.0526130
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0494310, upper bound: 0.0531694
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0494310, upper bound: 0.0574525
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0515243, upper bound: 0.0467692
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0512589, upper bound: 0.0467157
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0515243, upper bound: 0.0467692
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0512589, upper bound: 0.0467157
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0469424, upper bound: 0.0474952
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0469424, upper bound: 0.0477142
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0466613, upper bound: 0.0464569
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0466613, upper bound: 0.0465268
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0571108, upper bound: 0.0560705
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0583449, upper bound: 0.0577749
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0571108, upper bound: 0.0560705
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0583449, upper bound: 0.0577749
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0531691, upper bound: 0.0526445
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0529617, upper bound: 0.0525849
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0482200, upper bound: 0.0527781
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0482200, upper bound: 0.0483306
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0504925, upper bound: 0.0535316
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0504925, upper bound: 0.0535316
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0540103, upper bound: 0.0510472
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0540103, upper bound: 0.0510472
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0540103, upper bound: 0.0580074
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0540103, upper bound: 0.0580074
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0454287, upper bound: 0.0481373
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0454287, upper bound: 0.0518683
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0484318, upper bound: 0.0449764
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0525617, upper bound: 0.0477636
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0484318, upper bound: 0.0535126
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0525617, upper bound: 0.0560632
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0575010, upper bound: 0.0520676
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0597646, upper bound: 0.0538189
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0562547, upper bound: 0.0575275
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0562547, upper bound: 0.0611081
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0485324, upper bound: 0.0465314
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0530843, upper bound: 0.0501616
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0486342, upper bound: 0.0531497
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0486342, upper bound: 0.0585239
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0470584, upper bound: 0.0503109
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0457966, upper bound: 0.0495824
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0469319, upper bound: 0.0440349
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0522732, upper bound: 0.0477340
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0469319, upper bound: 0.0531018
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0522732, upper bound: 0.0560470
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0531476, upper bound: 0.0497296
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0578016, upper bound: 0.0527901
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0531476, upper bound: 0.0497296
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0578016, upper bound: 0.0527901
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0524593, upper bound: 0.0500621
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0567813, upper bound: 0.0516498
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0524593, upper bound: 0.0540369
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0567813, upper bound: 0.0551687
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0500831, upper bound: 0.0524888
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0500831, upper bound: 0.0524888
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0451840, upper bound: 0.0481373
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0494040, upper bound: 0.0518726
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0556762, upper bound: 0.0555834
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0583184, upper bound: 0.0581363
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0554189, upper bound: 0.0543604
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0580771, upper bound: 0.0567444
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0483161, upper bound: 0.0460313
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0537798, upper bound: 0.0477972
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0483161, upper bound: 0.0460312
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0537798, upper bound: 0.0477972
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0463466, upper bound: 0.0470902
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0463466, upper bound: 0.0475455
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0462078, upper bound: 0.0462078
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0462078, upper bound: 0.0463174
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0511345, upper bound: 0.0532298
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0511345, upper bound: 0.0573219
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0511345, upper bound: 0.0532298
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0511345, upper bound: 0.0484598
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0535281, upper bound: 0.0523795
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0532344, upper bound: 0.0523395
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0481364, upper bound: 0.0518787
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 8, lower bound: -0.0481364, upper bound: 0.0567152

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0008641, 0.0027994, -0.0007817, 0.0033154, -0.0041227, 0.0034623
1: -0.0027250, 0.0053399, -0.0023391, 0.0059787, -0.0087037, 0.0076790
2: 0.0089641, 0.0210083, 0.0083355, 0.0208251, -0.0118610, 0.0126260
3: -0.0062317, 0.0047288, -0.0083733, 0.0042942, -0.0105259, 0.0131022
4: -0.0079643, 0.0031963, -0.0084004, 0.0052278, -0.0131922, 0.0115967
5: -0.0023136, 0.0126584, -0.0070430, 0.0122245, -0.0145382, 0.0197014
6: 0.0030225, 0.0114011, 0.0015331, 0.0115792, -0.0085566, 0.0098681
7: -0.0258793, -0.0060774, -0.0249375, -0.0044513, -0.0211691, 0.0184567
8: 0.9496436, 1.0058178, 0.9523417, 1.0101700, -0.0605264, 0.0534760
9: -0.0032668, 0.0142959, -0.0041294, 0.0147099, -0.0177821, 0.0179050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0563027, upper bound: 0.0522557
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0563027, upper bound: 0.0522557
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0008289, 0.0032096, -0.0007828, 0.0034666, -0.0042497, 0.0038824
1: -0.0025602, 0.0058803, -0.0023444, 0.0061701, -0.0087304, 0.0082248
2: 0.0083522, 0.0209700, 0.0081278, 0.0209114, -0.0125593, 0.0128423
3: -0.0076462, 0.0045433, -0.0089043, 0.0043003, -0.0119464, 0.0134475
4: -0.0083888, 0.0043554, -0.0085444, 0.0057257, -0.0141145, 0.0128999
5: -0.0052153, 0.0124731, -0.0080932, 0.0122306, -0.0174459, 0.0205663
6: 0.0019117, 0.0115745, 0.0011283, 0.0116380, -0.0097264, 0.0104462
7: -0.0254772, -0.0047200, -0.0249506, -0.0039691, -0.0213588, 0.0197379
8: 0.9507955, 1.0095283, 0.9523044, 1.0114783, -0.0606828, 0.0572239
9: -0.0041065, 0.0145749, -0.0044144, 0.0149674, -0.0187209, 0.0186434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0554513, upper bound: 0.0515273
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0554513, upper bound: 0.0539938
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0007578, 0.0034337, -0.0025821, 0.0033348, -0.0040218, 0.0059480
1: -0.0022272, 0.0061330, -0.0035581, 0.0060450, -0.0082722, 0.0096910
2: 0.0081556, 0.0207358, 0.0081753, 0.0213194, -0.0131638, 0.0125604
3: -0.0087762, 0.0041683, -0.0081291, 0.0051295, -0.0139057, 0.0122974
4: -0.0085251, 0.0055381, -0.0085057, 0.0057372, -0.0142623, 0.0140438
5: -0.0078215, 0.0120988, -0.0062653, 0.0144801, -0.0223015, 0.0183642
6: 0.0012280, 0.0116301, 0.0014489, 0.0116207, -0.0103926, 0.0101813
7: -0.0246647, -0.0040655, -0.0267623, -0.0043102, -0.0203545, 0.0224504
8: 0.9531237, 1.0112284, 0.9490036, 1.0106113, -0.0574876, 0.0622249
9: -0.0043762, 0.0147183, -0.0043304, 0.0174938, -0.0213053, 0.0190486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0527701, upper bound: 0.0560721
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0548908, upper bound: 0.0575993
time: 1.36 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0009420, 0.0033028, -0.0025821, 0.0033348, -0.0041712, 0.0057576
1: -0.0026449, 0.0060034, -0.0035581, 0.0060450, -0.0086899, 0.0095614
2: 0.0082176, 0.0211313, 0.0081753, 0.0213194, -0.0130866, 0.0129560
3: -0.0079834, 0.0046500, -0.0081291, 0.0051295, -0.0131129, 0.0127792
4: -0.0084821, 0.0047281, -0.0085057, 0.0057372, -0.0142193, 0.0132338
5: -0.0059399, 0.0125683, -0.0062653, 0.0144801, -0.0204200, 0.0188336
6: 0.0016484, 0.0116126, 0.0014489, 0.0116207, -0.0099723, 0.0101637
7: -0.0256838, -0.0044123, -0.0267623, -0.0043102, -0.0210588, 0.0216651
8: 0.9502037, 1.0103606, 0.9490036, 1.0106113, -0.0604076, 0.0613570
9: -0.0042911, 0.0148685, -0.0043304, 0.0174938, -0.0209787, 0.0188441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0527701, upper bound: 0.0599950
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0548908, upper bound: 0.0607529
time: 1.24 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.81 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0563027, upper bound: 0.0522557
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0563027, upper bound: 0.0522557
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0554513, upper bound: 0.0515273
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0554513, upper bound: 0.0539938
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0527701, upper bound: 0.0560721
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0548908, upper bound: 0.0575993
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0527701, upper bound: 0.0599950
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0548908, upper bound: 0.0607529
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0501423, upper bound: 0.0488724
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0530275, upper bound: 0.0506864
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0506780, upper bound: 0.0542632
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0506780, upper bound: 0.0589097
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0487058, upper bound: 0.0515020
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0484687, upper bound: 0.0512766
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0492170, upper bound: 0.0467462
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0523427, upper bound: 0.0482045
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0492170, upper bound: 0.0539965
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0523427, upper bound: 0.0563308
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0559736, upper bound: 0.0522528
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0575259, upper bound: 0.0539796
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0603857, upper bound: 0.0600598
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0612295, upper bound: 0.0612295
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0490711, upper bound: 0.0482550
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0527025, upper bound: 0.0504740
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0490010, upper bound: 0.0539010
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0490010, upper bound: 0.0588564
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0533491, upper bound: 0.0539762
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0533491, upper bound: 0.0539762
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0532913, upper bound: 0.0538005
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0532913, upper bound: 0.0538005
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0514873, upper bound: 0.0476542
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0514420, upper bound: 0.0475900
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0487430, upper bound: 0.0525627
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0487430, upper bound: 0.0570115
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0491083, upper bound: 0.0468113
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0528051, upper bound: 0.0479876
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0491083, upper bound: 0.0468113
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0528051, upper bound: 0.0479876
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0515857, upper bound: 0.0478346
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0515857, upper bound: 0.0478346
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0510325, upper bound: 0.0539561
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0510325, upper bound: 0.0577429
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0510325, upper bound: 0.0486563
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0510325, upper bound: 0.0577429
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0534410, upper bound: 0.0526704
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0532431, upper bound: 0.0526130
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0494310, upper bound: 0.0531694
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0494310, upper bound: 0.0574525
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0515243, upper bound: 0.0467692
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0512589, upper bound: 0.0467157
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0515243, upper bound: 0.0467692
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0512589, upper bound: 0.0467157
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0571108, upper bound: 0.0560705
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0583449, upper bound: 0.0577749
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0571108, upper bound: 0.0560705
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0583449, upper bound: 0.0577749
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0531691, upper bound: 0.0526445
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0529617, upper bound: 0.0525849
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0482200, upper bound: 0.0527781
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0504925, upper bound: 0.0535316
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0504925, upper bound: 0.0535316
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0540103, upper bound: 0.0510472
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0540103, upper bound: 0.0510472
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0540103, upper bound: 0.0580074
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0540103, upper bound: 0.0580074
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0454287, upper bound: 0.0518683
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0525617, upper bound: 0.0477636
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0484318, upper bound: 0.0535126
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0525617, upper bound: 0.0560632
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0575010, upper bound: 0.0520676
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0597646, upper bound: 0.0538189
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0562547, upper bound: 0.0575275
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0562547, upper bound: 0.0611081
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0530843, upper bound: 0.0501616
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0486342, upper bound: 0.0531497
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0486342, upper bound: 0.0585239
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0470584, upper bound: 0.0503109
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0457966, upper bound: 0.0495824
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0522732, upper bound: 0.0477340
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0469319, upper bound: 0.0531018
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0522732, upper bound: 0.0560470
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0531476, upper bound: 0.0497296
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0578016, upper bound: 0.0527901
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0531476, upper bound: 0.0497296
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0578016, upper bound: 0.0527901
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0524593, upper bound: 0.0500621
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0567813, upper bound: 0.0516498
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0524593, upper bound: 0.0540369
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0567813, upper bound: 0.0551687
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0500831, upper bound: 0.0524888
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0500831, upper bound: 0.0524888
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0494040, upper bound: 0.0518726
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0556762, upper bound: 0.0555834
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0583184, upper bound: 0.0581363
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0554189, upper bound: 0.0543604
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0580771, upper bound: 0.0567444
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0537798, upper bound: 0.0477972
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0537798, upper bound: 0.0477972
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0511345, upper bound: 0.0532298
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0511345, upper bound: 0.0573219
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0511345, upper bound: 0.0532298
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0511345, upper bound: 0.0484598
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0535281, upper bound: 0.0523795
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0532344, upper bound: 0.0523395
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0481364, upper bound: 0.0518787
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0481364, upper bound: 0.0567152

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.82 + 597.94 = 601.76 seconds
