## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 17.859030126


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288)
1: (-45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655)
2: (-21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237)
3: (-39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091)
4: (-28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.30 + 1.84 = 3.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -17.9037896, upper bound: 17.9037896

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9036499, upper bound: 17.9037317
time: 0.67 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9037188, upper bound: 17.9037188
time: 0.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.41 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.41
Output dim: 0, lower bound: -17.9036499, upper bound: 17.9037317
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.41
Output dim: 0, lower bound: -17.9037188, upper bound: 17.9037188

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -10.5742512, 8.7518034, -11.2569275, 9.2909565, -19.8652058, 20.0087299
1: -41.5474434, 33.6830368, -44.2739563, 35.7098618, -77.2573013, 77.9569855
2: -20.0479450, 31.0254955, -21.3537350, 32.9903526, -53.0382996, 52.3792114
3: -36.3639297, 30.4327087, -38.7501068, 32.2506104, -68.6145401, 69.1828156
4: -26.6910076, 31.6035423, -28.4145012, 33.5703621, -60.2613678, 60.0180435

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9036499, upper bound: 17.9036499
time: 0.66 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9036499, upper bound: 17.9037188
time: 0.55 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -12.6187048, 10.4211016, -11.0338182, 9.1182871, -21.7369919, 21.4549179
1: -49.7551270, 40.3242188, -43.3985977, 35.0851974, -84.8403244, 83.7228165
2: -23.9273720, 36.7199593, -20.9132423, 32.3268433, -56.2542152, 57.6331978
3: -43.5699158, 36.3116417, -37.9861603, 31.6779575, -75.2478714, 74.2978058
4: -31.9433441, 37.4368324, -27.8598881, 32.9060974, -64.8494339, 65.2967224

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9037188, upper bound: 17.9036499
time: 0.61 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9037188, upper bound: 17.9037188
time: 0.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.63 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -17.9036499, upper bound: 17.9036499
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -17.9036499, upper bound: 17.9037188
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -17.9037188, upper bound: 17.9036499
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -17.9037188, upper bound: 17.9037188

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -10.5742512, 8.7518034, -10.5742512, 8.7518034, -19.3260536, 19.3260536
1: -41.5474434, 33.6830368, -41.5474434, 33.6830368, -75.2304611, 75.2304611
2: -20.0479450, 31.0254955, -20.0479450, 31.0254955, -51.0734329, 51.0734329
3: -36.3639297, 30.4327087, -36.3639297, 30.4327087, -66.7966385, 66.7966385
4: -26.6910076, 31.6035423, -26.6910076, 31.6035423, -58.2945480, 58.2945480

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8066585, upper bound: 17.8613184
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9025604, upper bound: 17.9025652
time: 0.59 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -10.5742512, 8.7518034, -12.6187048, 10.4211016, -20.9953518, 21.3705082
1: -41.5474434, 33.6830368, -49.7551270, 40.3242188, -81.8716507, 83.4381638
2: -20.0479450, 31.0254955, -23.9273720, 36.7199593, -56.7679062, 54.9528618
3: -36.3639297, 30.4327087, -43.5699158, 36.3116417, -72.6755676, 74.0026169
4: -26.6910076, 31.6035423, -31.9433441, 37.4368324, -64.1278381, 63.5468864

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8066585, upper bound: 17.8613184
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9025604, upper bound: 17.9029486
time: 0.70 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -12.6187048, 10.4211016, -10.5742512, 8.7518034, -21.3705082, 20.9953537
1: -49.7551270, 40.3242188, -41.5474434, 33.6830368, -83.4381638, 81.8716507
2: -23.9273720, 36.7199593, -20.0479450, 31.0254955, -54.9528656, 56.7679062
3: -43.5699158, 36.3116417, -36.3639297, 30.4327087, -74.0026169, 72.6755676
4: -31.9433441, 37.4368324, -26.6910076, 31.6035423, -63.5468864, 64.1278381

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9032621, upper bound: 17.9005991
time: 0.67 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9027147, upper bound: 17.9027147
time: 0.78 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -12.6187048, 10.4211016, -12.6187048, 10.4211016, -23.0398045, 23.0398064
1: -49.7551270, 40.3242188, -49.7551270, 40.3242188, -90.0793457, 90.0793457
2: -23.9273720, 36.7199593, -23.9273720, 36.7199593, -60.6473312, 60.6473312
3: -43.5699158, 36.3116417, -43.5699158, 36.3116417, -79.8815613, 79.8815613
4: -31.9433441, 37.4368324, -31.9433441, 37.4368324, -69.3801727, 69.3801727

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9005991, upper bound: 17.9027059
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9033535, upper bound: 17.9033104
time: 0.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.85 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -17.8066585, upper bound: 17.8613184
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -17.9025604, upper bound: 17.9025652
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -17.8066585, upper bound: 17.8613184
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -17.9025604, upper bound: 17.9029486
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -17.9032621, upper bound: 17.9005991
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -17.9027147, upper bound: 17.9027147
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -17.9005991, upper bound: 17.9027059
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -17.9033535, upper bound: 17.9033104

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.6250286, 7.2796216, -9.8935013, 8.2299433, -16.8549709, 17.1731167
1: -33.8804512, 28.4254265, -38.8650513, 31.7836742, -65.6641235, 67.2904816
2: -16.3679333, 25.3747883, -18.7530479, 29.0417633, -45.4096909, 44.1278381
3: -29.5793114, 25.6486206, -33.9972458, 28.7091827, -58.2884941, 59.6458664
4: -21.7532978, 26.2463226, -24.9625626, 29.7224369, -51.4757347, 51.2088814

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8382349, upper bound: 17.8382349
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8382349, upper bound: 17.8871196
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -9.7539425, 8.1305418, -10.4785004, 8.6779270, -18.4318695, 18.6090431
1: -38.2927933, 31.4316177, -41.1700439, 33.4136848, -71.7064819, 72.6016617
2: -18.5312443, 28.6760941, -19.8687096, 30.7408028, -49.2720451, 48.5448036
3: -33.4824905, 28.3932915, -36.0314178, 30.1872673, -63.6697578, 64.4247131
4: -24.6021042, 29.3805351, -26.4491863, 31.3357563, -55.9378586, 55.8297195

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8871196, upper bound: 17.8396372
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8871196, upper bound: 17.9025652
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.6250286, 7.2796216, -11.9186163, 9.8824596, -18.5074883, 19.1982346
1: -33.8804512, 28.4254265, -46.9846916, 38.3486404, -72.2290955, 75.4101105
2: -16.3679333, 25.3747883, -22.6099319, 34.6597023, -51.0276299, 47.9847183
3: -29.5793114, 25.6486206, -41.1264687, 34.5229378, -64.1022415, 66.7750778
4: -21.7532978, 26.2463226, -30.1725368, 35.4881516, -57.2414474, 56.4188538

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7868409, upper bound: 17.8440922
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8060390, upper bound: 17.8612303
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -9.7539425, 8.1305418, -12.5177431, 10.3427515, -20.0966930, 20.6482849
1: -38.2927933, 31.4316177, -49.3552361, 40.0383110, -78.3310928, 80.7868500
2: -18.5312443, 28.6760941, -23.7369289, 36.4209900, -54.9522324, 52.4130249
3: -33.4824905, 28.3932915, -43.2183266, 36.0517006, -69.5341797, 71.6116180
4: -24.6021042, 29.3805351, -31.6887512, 37.1527863, -61.7548904, 61.0692787

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9025604, upper bound: 17.9029486
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9025604, upper bound: 17.9025908
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -12.1592789, 10.0566502, -9.0323858, 7.5845494, -19.7438278, 19.0890350
1: -47.9199448, 38.9341469, -35.3971825, 29.4503365, -77.3702850, 74.3313293
2: -23.0697918, 35.4404640, -17.1530113, 26.6203766, -49.6901703, 52.5934753
3: -41.9587746, 35.0696716, -30.9263210, 26.6152000, -68.5739670, 65.9959946
4: -30.7725887, 36.1497154, -22.7944241, 27.4255714, -58.1981583, 58.9441376

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9033533, upper bound: 17.9025777
time: 0.77 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9027290, upper bound: 17.9025777
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -12.5857544, 10.3949823, -10.2785854, 8.5181694, -21.1039238, 20.6735668
1: -49.6240540, 40.2261200, -40.3782425, 32.8089638, -82.4330139, 80.6043625
2: -23.8653297, 36.6257439, -19.4826870, 30.1687393, -54.0340691, 56.1084290
3: -43.4551086, 36.2234459, -35.3370857, 29.6414185, -73.0965195, 71.5605240
4: -31.8595924, 37.3436852, -25.9385757, 30.7632217, -62.6228065, 63.2822609

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8955356, upper bound: 17.9006432
time: 0.77 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9032145, upper bound: 17.9026886
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -10.7873058, 9.0056572, -12.1592789, 10.0566502, -20.8439560, 21.1649323
1: -42.4250641, 35.0572472, -47.9199448, 38.9341469, -81.3591995, 82.9771881
2: -20.5561352, 31.4658680, -23.0697918, 35.4404640, -55.9965973, 54.5356560
3: -37.1309280, 31.5597267, -41.9587746, 35.0696716, -72.2005997, 73.5185013
4: -27.3118782, 32.3669090, -30.7725887, 36.1497154, -63.4615936, 63.1394920

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9005321, upper bound: 17.9005321
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9005321, upper bound: 17.9027059
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -12.2423887, 10.1220493, -12.5857544, 10.3949823, -22.6373711, 22.7077980
1: -48.2582016, 39.2014313, -49.6240540, 40.2261200, -88.4843216, 88.8254852
2: -23.2183285, 35.6403351, -23.8653297, 36.6257439, -59.8440628, 59.5056648
3: -42.2585106, 35.3018341, -43.4551086, 36.2234459, -78.4819489, 78.7569427
4: -30.9872990, 36.3691940, -31.8595924, 37.3436852, -68.3309860, 68.2287598

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9032621, upper bound: 17.9005991
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9032621, upper bound: 17.9033104
time: 0.74 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.78 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 0, lower bound: -17.8382349, upper bound: 17.8382349
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -17.8382349, upper bound: 17.8871196
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -17.8871196, upper bound: 17.8396372
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -17.8871196, upper bound: 17.9025652
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 0, lower bound: -17.7868409, upper bound: 17.8440922
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -17.8060390, upper bound: 17.8612303
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -17.9025604, upper bound: 17.9029486
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -17.9025604, upper bound: 17.9025908
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -17.9033533, upper bound: 17.9025777
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -17.9027290, upper bound: 17.9025777
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -17.8955356, upper bound: 17.9006432
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -17.9032145, upper bound: 17.9026886
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -17.9005321, upper bound: 17.9005321
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -17.9005321, upper bound: 17.9027059
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -17.9032621, upper bound: 17.9005991
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -17.9032621, upper bound: 17.9033104

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8.6250286, 7.2796216, -9.7539425, 8.1305418, -16.7555695, 17.0335636
1: -33.8804512, 28.4254265, -38.2927933, 31.4316177, -65.3120728, 66.7182159
2: -16.3679333, 25.3747883, -18.5312443, 28.6760941, -45.0440254, 43.9060326
3: -29.5793114, 25.6486206, -33.4824905, 28.3932915, -57.9725990, 59.1311111
4: -21.7532978, 26.2463226, -24.6021042, 29.3805351, -51.1338348, 50.8484230

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8334996, upper bound: 17.8724704
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8381519, upper bound: 17.8871196
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -9.7539425, 8.1305418, -8.6250286, 7.2796216, -17.0335636, 16.7555695
1: -38.2927933, 31.4316177, -33.8804512, 28.4254265, -66.7182159, 65.3120651
2: -18.5312443, 28.6760941, -16.3679333, 25.3747883, -43.9060326, 45.0440254
3: -33.4824905, 28.3932915, -29.5793114, 25.6486206, -59.1311111, 57.9726028
4: -24.6021042, 29.3805351, -21.7532978, 26.2463226, -50.8484230, 51.1338348

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8724704, upper bound: 17.8335947
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8871196, upper bound: 17.8395542
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -9.7539425, 8.1305418, -9.7539425, 8.1305418, -17.8844833, 17.8844833
1: -38.2927933, 31.4316177, -38.2927933, 31.4316177, -69.7244110, 69.7244110
2: -18.5312443, 28.6760941, -18.5312443, 28.6760941, -47.2073364, 47.2073364
3: -33.4824905, 28.3932915, -33.4824905, 28.3932915, -61.8757820, 61.8757744
4: -24.6021042, 29.3805351, -24.6021042, 29.3805351, -53.9826355, 53.9826393

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8724704, upper bound: 17.9025536
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8871196, upper bound: 17.9025618
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8.5469522, 7.2167897, -11.5883141, 9.6088705, -18.1558189, 18.8051014
1: -33.5701790, 28.1857872, -45.6768303, 37.3100204, -70.8802032, 73.8626175
2: -16.2221832, 25.1492786, -21.9812222, 33.6552353, -49.8774109, 47.1305008
3: -29.3078709, 25.4352798, -39.9858170, 33.5834160, -62.8912811, 65.4210892
4: -21.5551701, 26.0212154, -29.3383026, 34.4965553, -56.0517235, 55.3595200

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7377314, upper bound: 17.8232672
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7378774, upper bound: 17.8302623
time: 1.17 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -9.4994354, 7.9299378, -11.3675461, 9.4429636, -18.9423981, 19.2974834
1: -37.2900734, 30.6900921, -44.7939682, 36.6583138, -73.9483871, 75.4840317
2: -18.0460300, 27.9401608, -21.5455952, 33.0754700, -51.1214867, 49.4857559
3: -32.5960732, 27.7284794, -39.2080536, 33.0442200, -65.6402893, 66.9365311
4: -23.9559288, 28.6693268, -28.7734127, 33.9424591, -57.8983879, 57.4427414

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8996655, upper bound: 17.8906651
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9025370, upper bound: 17.9028269
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -9.2632408, 7.7350159, -11.5117035, 9.5576010, -18.8208408, 19.2467175
1: -36.3472061, 29.8878441, -45.2462997, 37.1195145, -73.4667206, 75.1341400
2: -17.5886250, 27.2794590, -21.9861832, 33.4620094, -51.0506248, 49.2656403
3: -31.7691154, 27.0272675, -39.5393600, 33.4493294, -65.2184448, 66.5666275
4: -23.3445129, 27.9952698, -29.1155262, 34.4167519, -57.7612610, 57.1107864

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8951279, upper bound: 17.8818501
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9025371, upper bound: 17.9025730
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -11.0334330, 9.1782780, -8.7744780, 7.3861303, -18.4195633, 17.9527550
1: -43.4621544, 35.6396179, -34.3798866, 28.7088242, -72.1709747, 70.0195007
2: -20.9213505, 32.1606331, -16.6665478, 25.8752308, -46.7965775, 48.8271790
3: -38.0473785, 32.1314011, -30.0343933, 25.9500504, -63.9974289, 62.1657944
4: -27.9253044, 33.0037460, -22.1434727, 26.7122040, -54.6375046, 55.1472168

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8902405, upper bound: 17.9000709
time: 1.50 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9032223, upper bound: 17.9025516
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -11.1470642, 9.2718821, -8.6340694, 7.2680640, -18.4151287, 17.9059505
1: -43.7935562, 36.0258827, -33.8305435, 28.2309380, -72.0244904, 69.8564301
2: -21.2921219, 32.4587784, -16.3852520, 25.4819965, -46.7741165, 48.8440323
3: -38.2869797, 32.4580193, -29.5500069, 25.5325413, -63.8195076, 62.0080223
4: -28.1976013, 33.3894997, -21.7814007, 26.3003483, -54.4979477, 55.1708984

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8801928, upper bound: 17.8939436
time: 0.60 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9027135, upper bound: 17.9025517
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -11.9186525, 9.8448524, -10.0810966, 8.3592663, -20.2779160, 19.9259472
1: -46.9957275, 38.1545601, -39.5986328, 32.2241936, -79.2199249, 77.7531891
2: -22.6049786, 34.6148376, -19.1060905, 29.5840912, -52.1890717, 53.7209282
3: -41.1410675, 34.3606491, -34.6539841, 29.1160488, -70.2571182, 69.0146332
4: -30.1675205, 35.3594284, -25.4365807, 30.1934261, -60.3609467, 60.7960091

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8409262, upper bound: 17.8941187
time: 0.68 seconds

## Relational analysis of NS_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8270830, upper bound: 17.8468819
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -12.2205286, 10.0956755, -10.1853962, 8.4422131, -20.6627331, 20.2810688
1: -48.1689911, 39.0928688, -40.0088005, 32.5176201, -80.6866150, 79.1016617
2: -23.1780739, 35.5341454, -19.3061104, 29.8870392, -53.0651131, 54.8402519
3: -42.1828308, 35.2022476, -35.0155640, 29.3789234, -71.5617523, 70.2177963
4: -30.9360409, 36.2651787, -25.7030373, 30.4870071, -61.4230423, 61.9682159

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9026780, upper bound: 17.9025268
time: 0.70 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9026866, upper bound: 17.9026554
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -10.7873058, 9.0056572, -10.7873058, 9.0056572, -19.7929611, 19.7929611
1: -42.4250641, 35.0572472, -42.4250641, 35.0572472, -77.4823151, 77.4823151
2: -20.5561352, 31.4658680, -20.5561352, 31.4658680, -52.0219955, 52.0219955
3: -37.1309280, 31.5597267, -37.1309280, 31.5597267, -68.6906509, 68.6906586
4: -27.3118782, 32.3669090, -27.3118782, 32.3669090, -59.6787834, 59.6787834

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9004988, upper bound: 17.8993661
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9005123, upper bound: 17.9005123
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -10.7873058, 9.0056572, -12.2423887, 10.1220493, -20.9093533, 21.2480469
1: -42.4250641, 35.0572472, -48.2582016, 39.2014313, -81.6264954, 83.3154449
2: -20.5561352, 31.4658680, -23.2183285, 35.6403351, -56.1964684, 54.6841812
3: -37.1309280, 31.5597267, -42.2585106, 35.3018341, -72.4327621, 73.8182220
4: -27.3118782, 32.3669090, -30.9872990, 36.3691940, -63.6810722, 63.3542099

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9004988, upper bound: 17.9024226
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9004988, upper bound: 17.9026673
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -12.2423887, 10.1220493, -10.7873058, 9.0056572, -21.2480450, 20.9093533
1: -48.2582016, 39.2014313, -42.4250641, 35.0572472, -83.3154449, 81.6264954
2: -23.2183285, 35.6403351, -20.5561352, 31.4658680, -54.6841812, 56.1964684
3: -42.2585106, 35.3018341, -37.1309280, 31.5597267, -73.8182220, 72.4327545
4: -30.9872990, 36.3691940, -27.3118782, 32.3669090, -63.3542099, 63.6810570

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9029557, upper bound: 17.9005789
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9032399, upper bound: 17.9005694
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -12.2423887, 10.1220493, -12.2423887, 10.1220493, -22.3644371, 22.3644371
1: -48.2582016, 39.2014313, -48.2582016, 39.2014313, -87.4596329, 87.4596329
2: -23.2183285, 35.6403351, -23.2183285, 35.6403351, -58.8586578, 58.8586578
3: -42.2585106, 35.3018341, -42.2585106, 35.3018341, -77.5603333, 77.5603333
4: -30.9872990, 36.3691940, -30.9872990, 36.3691940, -67.3564911, 67.3564911

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9029557, upper bound: 17.9032902
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9005123, upper bound: 17.9032892
time: 0.76 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.27 seconds
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.8334996, upper bound: 17.8724704
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.8381519, upper bound: 17.8871196
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.8724704, upper bound: 17.8335947
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.8871196, upper bound: 17.8395542
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.8724704, upper bound: 17.9025536
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.8871196, upper bound: 17.9025618
NS_A1_B2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.7377314, upper bound: 17.8232672
NS_A1_B2_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.7378774, upper bound: 17.8302623
NS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.8996655, upper bound: 17.8906651
NS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.9025370, upper bound: 17.9028269
NS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.8951279, upper bound: 17.8818501
NS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.9025371, upper bound: 17.9025730
NS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.8902405, upper bound: 17.9000709
NS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.9032223, upper bound: 17.9025516
NS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.8801928, upper bound: 17.8939436
NS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.9027135, upper bound: 17.9025517
NS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.8409262, upper bound: 17.8941187
NS_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.8270830, upper bound: 17.8468819
NS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.9026780, upper bound: 17.9025268
NS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.9026866, upper bound: 17.9026554
NS_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.9004988, upper bound: 17.8993661
NS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.9005123, upper bound: 17.9005123
NS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.9004988, upper bound: 17.9024226
NS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.9004988, upper bound: 17.9026673
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.9029557, upper bound: 17.9005789
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.9032399, upper bound: 17.9005694
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.9029557, upper bound: 17.9032902
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -17.9005123, upper bound: 17.9032892

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -8.3355255, 7.0549717, -8.7259398, 7.3241320, -15.6596575, 15.7809114
1: -32.7411232, 27.5941105, -34.2524223, 28.4322987, -61.1734123, 61.8465347
2: -15.8111601, 24.5274620, -16.5749817, 25.7034645, -41.5146255, 41.1024437
3: -28.5736370, 24.9009838, -29.9225197, 25.7123299, -54.2859650, 54.8235016
4: -21.0188713, 25.4416027, -21.9909878, 26.5092525, -47.5281181, 47.4325905

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8138516, upper bound: 17.8491261
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8138516, upper bound: 17.8724535
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -8.2441349, 6.9731975, -8.8429289, 7.4126472, -15.6567822, 15.8161259
1: -32.3620644, 27.2164497, -34.6158562, 28.8365402, -61.1986046, 61.8322945
2: -15.6506424, 24.3140106, -16.8528938, 25.9942722, -41.6449089, 41.1669006
3: -28.2417622, 24.5856915, -30.1786442, 26.1047440, -54.3465042, 54.7643356
4: -20.7711697, 25.1798096, -22.2614460, 26.9329090, -47.7040787, 47.4412537

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7562091, upper bound: 17.8451536
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8395346, upper bound: 17.8871168
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8.7259398, 7.3241320, -8.3355255, 7.0549717, -15.7809114, 15.6596575
1: -34.2524223, 28.4322987, -32.7411232, 27.5941105, -61.8465347, 61.1734085
2: -16.5749817, 25.7034645, -15.8111601, 24.5274620, -41.1024437, 41.5146255
3: -29.9225197, 25.7123299, -28.5736370, 24.9009838, -54.8235016, 54.2859650
4: -21.9909878, 26.5092525, -21.0188713, 25.4416027, -47.4325905, 47.5281219

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8491261, upper bound: 17.8138516
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8724535, upper bound: 17.8335732
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.8429289, 7.4126472, -8.2441349, 6.9731975, -15.8161259, 15.6567822
1: -34.6158562, 28.8365402, -32.3620644, 27.2164497, -61.8322945, 61.1986046
2: -16.8528938, 25.9942722, -15.6506424, 24.3140106, -41.1669006, 41.6449089
3: -30.1786442, 26.1047440, -28.2417622, 24.5856915, -54.7643356, 54.3465042
4: -22.2614460, 26.9329090, -20.7711697, 25.1798096, -47.4412537, 47.7040787

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8122421, upper bound: 17.7562091
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8871168, upper bound: 17.8395346
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.7259398, 7.3241320, -9.4994354, 7.9299378, -16.6558781, 16.8235664
1: -34.2524223, 28.4322987, -37.2900734, 30.6900921, -64.9425125, 65.7223663
2: -16.5749817, 25.7034645, -18.0460300, 27.9401608, -44.5151443, 43.7494965
3: -29.9225197, 25.7123299, -32.5960732, 27.7284794, -57.6509972, 58.3083992
4: -21.9909878, 26.5092525, -23.9559288, 28.6693268, -50.6603088, 50.4651794

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9022864, upper bound: 17.9023107
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9022864, upper bound: 17.9025536
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.8429289, 7.4126472, -9.2632408, 7.7350159, -16.5779419, 16.6758881
1: -34.6158562, 28.8365402, -36.3472061, 29.8878441, -64.5036850, 65.1837463
2: -16.8528938, 25.9942722, -17.5886250, 27.2794590, -44.1323509, 43.5828857
3: -30.1786442, 26.1047440, -31.7691154, 27.0272675, -57.2059097, 57.8738556
4: -22.2614460, 26.9329090, -23.3445129, 27.9952698, -50.2567139, 50.2774200

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8951269, upper bound: 17.8710772
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9025476, upper bound: 17.9025390
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -9.3133097, 7.7806449, -10.7475958, 8.9287844, -18.2420921, 18.5282402
1: -36.5557861, 30.1419563, -42.3722610, 34.7116814, -71.2674713, 72.5142136
2: -17.6923733, 27.3982162, -20.3644257, 31.1756859, -48.8680573, 47.7626381
3: -31.9500675, 27.2393589, -37.0851059, 31.2882519, -63.2383194, 64.3244629
4: -23.4818497, 28.1369915, -27.2114830, 32.0834084, -55.5652580, 55.3484726

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8663581, upper bound: 17.8762392
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8890277, upper bound: 17.8278493
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8234443, upper bound: 17.8247537
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -9.4136782, 7.8612294, -11.0125475, 9.1496983, -18.5633774, 18.8737755
1: -36.9482918, 30.4271336, -43.3859024, 35.5455666, -72.4938583, 73.8130341
2: -17.8857651, 27.6915092, -20.8706188, 31.9972553, -49.8830147, 48.5621262
3: -32.2970314, 27.4932098, -37.9809380, 32.0357628, -64.3327942, 65.4741516
4: -23.7384224, 28.4226837, -27.8806610, 32.8811302, -56.6195488, 56.3033371

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8638921, upper bound: 17.8776480
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9022941, upper bound: 17.9028131
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9022941, upper bound: 17.9028269
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -9.0755358, 7.5852280, -10.8463573, 9.0178776, -18.0934143, 18.4315834
1: -35.6051140, 29.3377228, -42.6392288, 35.0702133, -70.6753235, 71.9769516
2: -17.2318974, 26.7321434, -20.7270355, 31.4454079, -48.6772957, 47.4591789
3: -31.1163197, 26.5345650, -37.2531128, 31.6018181, -62.7181396, 63.7876663
4: -22.8663864, 27.4614849, -27.4359016, 32.4577560, -55.3241425, 54.8973732

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8759749, upper bound: 17.8202360
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8232126, upper bound: 17.8185355
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -9.1821337, 7.6703482, -11.1822262, 9.2836943, -18.4658260, 18.8525734
1: -36.0242271, 29.6396999, -43.9437981, 36.0738144, -72.0980377, 73.5834885
2: -17.4375763, 27.0454102, -21.3624935, 32.4593239, -49.8969002, 48.4079056
3: -31.4872398, 26.8040504, -38.4021912, 32.5083466, -63.9955864, 65.2062378
4: -23.1389847, 27.7597027, -28.2842236, 33.4302063, -56.5691910, 56.0439262

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9024299, upper bound: 17.9015409
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9024233, upper bound: 17.9024604
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -10.3989506, 8.6526699, -8.6018534, 7.2524686, -17.6514187, 17.2545223
1: -40.9873886, 33.6513023, -33.6980705, 28.2095795, -69.1969452, 67.3493729
2: -19.7128849, 30.2137794, -16.3448696, 25.3748875, -45.0877724, 46.5586319
3: -35.8751602, 30.3377705, -29.4360085, 25.5035419, -61.3787003, 59.7737808
4: -26.3270168, 31.1012478, -21.7054958, 26.2356644, -52.5626793, 52.8067360

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8391616, upper bound: 17.8915714
time: 0.73 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8270830, upper bound: 17.8238706
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -10.6875019, 8.8927183, -8.6879387, 7.3166089, -18.0041103, 17.5806580
1: -42.0920639, 34.5548553, -34.0358810, 28.4438610, -70.5359268, 68.5907364
2: -20.2635574, 31.1094837, -16.5049496, 25.6245022, -45.8880539, 47.6144257
3: -36.8536072, 31.1483784, -29.7338486, 25.7123146, -62.5659218, 60.8822250
4: -27.0549927, 31.9705544, -21.9238071, 26.4647331, -53.5197220, 53.8943634

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8418255, upper bound: 17.8916058
time: 0.59 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8300784, upper bound: 17.8241645
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -10.4746599, 8.7253218, -8.4616632, 7.1335506, -17.6082077, 17.1869812
1: -41.1591873, 33.9492378, -33.1493683, 27.7300358, -68.8892136, 67.0986023
2: -20.0162258, 30.4127293, -16.0604839, 24.9817924, -44.9980164, 46.4732132
3: -35.9647484, 30.5859985, -28.9523811, 25.0870609, -61.0518112, 59.5383797
4: -26.4991074, 31.4060669, -21.3438892, 25.8211842, -52.3202896, 52.7499542

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8268666, upper bound: 17.8760826
time: 0.65 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8251578, upper bound: 17.8219052
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -10.8330154, 9.0090237, -8.5490284, 7.1993895, -18.0324039, 17.5580482
1: -42.5528641, 35.0202827, -33.4922295, 27.9689007, -70.5217514, 68.5124969
2: -20.6958809, 31.4976101, -16.2260056, 25.2355251, -45.9314041, 47.7236099
3: -37.1940002, 31.5516186, -29.2543049, 25.2971935, -62.4911957, 60.8059235
4: -27.4040337, 32.4440689, -21.5655136, 26.0564804, -53.4605141, 54.0095825

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8275611, upper bound: 17.8761932
time: 0.66 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8258590, upper bound: 17.8221237
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -11.8622503, 9.7991972, -10.0378838, 8.3385048, -20.2007504, 19.8370781
1: -46.7738380, 37.9806824, -39.4474487, 32.1964798, -78.9703140, 77.4281311
2: -22.4974442, 34.4498634, -18.9900875, 29.3826313, -51.8800697, 53.4399452
3: -40.9470863, 34.2035904, -34.5526505, 29.0549469, -70.0020294, 68.7562408
4: -30.0254002, 35.1934891, -25.3581505, 30.0641556, -60.0895462, 60.5516396

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8263854, upper bound: 17.8938613
time: 1.51 seconds

## Relational analysis of NS_A2_B1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8313567, upper bound: 17.8932189
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -11.9313383, 9.8691988, -9.1237316, 7.6105037, -19.5418415, 18.9929314
1: -47.0229492, 38.2439423, -35.8244705, 29.4282837, -76.4512329, 74.0683975
2: -22.6256485, 34.6956291, -17.2819672, 26.8210735, -49.4467239, 51.9775963
3: -41.1789093, 34.4451714, -31.3372097, 26.6123447, -67.7912369, 65.7823792
4: -30.2052059, 35.4576149, -23.0079842, 27.5286427, -57.7338486, 58.4655914

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9023810, upper bound: 17.9025124
time: 0.94 seconds

## Relational analysis of NS_A2_B1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9026530, upper bound: 17.9025025
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -11.8548975, 9.8069372, -9.2189779, 7.6853681, -19.5402641, 19.0259113
1: -46.7082138, 37.9841995, -36.1127853, 29.7685261, -76.4767380, 74.0969849
2: -22.4953213, 34.5230827, -17.5332279, 27.0893955, -49.5847168, 52.0563087
3: -40.8964310, 34.2222366, -31.5444679, 26.9403172, -67.8367462, 65.7667084
4: -29.9959507, 35.2587433, -23.2400360, 27.8891563, -57.8851089, 58.4987717

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9023892, upper bound: 17.9026409
time: 0.73 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9026611, upper bound: 17.9026310
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -10.7511530, 8.9777994, -10.4891205, 8.7764101, -19.5275631, 19.4669189
1: -42.2807541, 34.9547653, -41.2323914, 34.2129707, -76.4937286, 76.1871490
2: -20.4838772, 31.3640785, -19.9594765, 30.6336441, -51.1175194, 51.3235550
3: -37.0032692, 31.4678993, -36.0731812, 30.8062382, -67.8095093, 67.5410690
4: -27.2199802, 32.2685738, -26.5507507, 31.5599766, -58.7799568, 58.8193245

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8993527, upper bound: 17.8993527
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8993527, upper bound: 17.8993661
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -10.4223270, 8.7118139, -10.8613405, 9.0572186, -19.4795456, 19.5731525
1: -40.9955673, 33.9691963, -42.7690125, 35.4195366, -76.4150925, 76.7382050
2: -19.8887463, 30.3656635, -20.7385616, 31.4904613, -51.3792076, 51.1042252
3: -35.8738708, 30.5747852, -37.4092827, 31.8676853, -67.7415543, 67.9840622
4: -26.3911457, 31.3166409, -27.5132542, 32.5443573, -58.9355011, 58.8298950

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8993661, upper bound: 17.9004988
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8993661, upper bound: 17.9005123
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -10.7511530, 8.9777994, -11.9412880, 9.8914385, -20.6425915, 20.9190865
1: -42.2807541, 34.9547653, -47.0488510, 38.3466263, -80.6273651, 82.0036163
2: -20.4838772, 31.3640785, -22.6441879, 34.8102417, -55.2941208, 54.0082588
3: -37.0032692, 31.4678993, -41.1872787, 34.5389404, -71.5421982, 72.6551743
4: -27.2199802, 32.2685738, -30.2141342, 35.5589371, -62.7789154, 62.4827042

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8994230, upper bound: 17.9024226
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8994230, upper bound: 17.9024226
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -10.4223270, 8.7118139, -12.0305853, 9.9370108, -20.3593369, 20.7423992
1: -40.9955673, 33.9691963, -47.4764481, 38.6831589, -79.6787262, 81.4456482
2: -19.8887463, 30.3656635, -22.8127365, 34.8363533, -54.7250977, 53.1783981
3: -35.8738708, 30.5747852, -41.5374870, 34.8268051, -70.7006760, 72.1122742
4: -26.3911457, 31.3166409, -30.4614754, 35.6952209, -62.0863647, 61.7781143

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8994230, upper bound: 17.9026673
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8994230, upper bound: 17.9026673
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -11.9412880, 9.8914385, -10.7511530, 8.9777994, -20.9190865, 20.6425915
1: -47.0488510, 38.3466263, -42.2807541, 34.9547653, -82.0036163, 80.6273651
2: -22.6441879, 34.8102417, -20.4838772, 31.3640785, -54.0082588, 55.2941208
3: -41.1872787, 34.5389404, -37.0032692, 31.4678993, -72.6551819, 71.5421982
4: -30.2141342, 35.5589371, -27.2199802, 32.2685738, -62.4827042, 62.7789154

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9029517, upper bound: 17.8994232
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9029517, upper bound: 17.9005694
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -12.0305853, 9.9370108, -10.4223270, 8.7118139, -20.7423992, 20.3593369
1: -47.4764481, 38.6831589, -40.9955673, 33.9691963, -81.4456482, 79.6787262
2: -22.8127365, 34.8363533, -19.8887463, 30.3656635, -53.1783981, 54.7250977
3: -41.5374870, 34.8268051, -35.8738708, 30.5747852, -72.1122665, 70.7006760
4: -30.4614754, 35.6952209, -26.3911457, 31.3166409, -61.7781143, 62.0863647

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9032066, upper bound: 17.8994232
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9032066, upper bound: 17.9005694
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -11.9412880, 9.8914385, -12.2073116, 10.0953321, -22.0366192, 22.0987492
1: -47.0488510, 38.3466263, -48.1179466, 39.1023483, -86.1511993, 86.4645462
2: -22.6441879, 34.8102417, -23.1517353, 35.5440598, -58.1882401, 57.9619751
3: -41.1872787, 34.5389404, -42.1343765, 35.2134361, -76.4007111, 76.6733093
4: -30.2141342, 35.5589371, -30.8975639, 36.2752495, -66.4893799, 66.4564896

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9030214, upper bound: 17.9030070
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9030214, upper bound: 17.9032892
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -12.0305853, 9.9370108, -11.7828321, 9.7525949, -21.7831802, 21.7198429
1: -47.4764481, 38.6831589, -46.4522972, 37.8368034, -85.3132401, 85.1354523
2: -22.8127365, 34.8363533, -22.3429890, 34.2619972, -57.0747337, 57.1793327
3: -41.5374870, 34.8268051, -40.6704025, 34.0665131, -75.6040039, 75.4972076
4: -30.4614754, 35.6952209, -29.8296013, 35.0403252, -65.5017929, 65.5248184

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9033211, upper bound: 17.9030070
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9004988, upper bound: 17.9032892
time: 0.73 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.55 seconds
NS_A1_B1_A1_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8138516, upper bound: 17.8491261
NS_A1_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8138516, upper bound: 17.8724535
NS_A1_B1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.7562091, upper bound: 17.8451536
NS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8395346, upper bound: 17.8871168
NS_A1_B1_A2_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8491261, upper bound: 17.8138516
NS_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8724535, upper bound: 17.8335732
NS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8122421, upper bound: 17.7562091
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8871168, upper bound: 17.8395346
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9022864, upper bound: 17.9023107
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9022864, upper bound: 17.9025536
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8951269, upper bound: 17.8710772
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9025476, upper bound: 17.9025390
NS_A1_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8890277, upper bound: 17.8278493
NS_A1_B2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8234443, upper bound: 17.8247537
NS_A1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9022941, upper bound: 17.9028131
NS_A1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9022941, upper bound: 17.9028269
NS_A1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8759749, upper bound: 17.8202360
NS_A1_B2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8232126, upper bound: 17.8185355
NS_A1_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9024299, upper bound: 17.9015409
NS_A1_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9024233, upper bound: 17.9024604
NS_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8391616, upper bound: 17.8915714
NS_A2_B1_B1_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8270830, upper bound: 17.8238706
NS_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8418255, upper bound: 17.8916058
NS_A2_B1_B1_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8300784, upper bound: 17.8241645
NS_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8268666, upper bound: 17.8760826
NS_A2_B1_B1_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8251578, upper bound: 17.8219052
NS_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8275611, upper bound: 17.8761932
NS_A2_B1_B1_A2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8258590, upper bound: 17.8221237
NS_A2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8263854, upper bound: 17.8938613
NS_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8313567, upper bound: 17.8932189
NS_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9023810, upper bound: 17.9025124
NS_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9026530, upper bound: 17.9025025
NS_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9023892, upper bound: 17.9026409
NS_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9026611, upper bound: 17.9026310
NS_A2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8993527, upper bound: 17.8993527
NS_A2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8993527, upper bound: 17.8993661
NS_A2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8993661, upper bound: 17.9004988
NS_A2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8993661, upper bound: 17.9005123
NS_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8994230, upper bound: 17.9024226
NS_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8994230, upper bound: 17.9024226
NS_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8994230, upper bound: 17.9026673
NS_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.8994230, upper bound: 17.9026673
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9029517, upper bound: 17.8994232
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9029517, upper bound: 17.9005694
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9032066, upper bound: 17.8994232
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9032066, upper bound: 17.9005694
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9030214, upper bound: 17.9030070
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9030214, upper bound: 17.9032892
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9033211, upper bound: 17.9030070
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -17.9004988, upper bound: 17.9032892

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -8.2565022, 6.9908934, -8.4286814, 7.0882740, -15.3447742, 15.4195738
1: -32.4270477, 27.3520432, -33.0687981, 27.5342007, -59.9612465, 60.4208412
2: -15.6635942, 24.2992191, -16.0192089, 24.8453274, -40.5089188, 40.3184242
3: -28.2990189, 24.6859169, -28.8889580, 24.9061794, -53.2052002, 53.5748749
4: -20.8182182, 25.2143879, -21.2380276, 25.6587963, -46.4770126, 46.4524155

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8065985, upper bound: 17.8690294
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8065985, upper bound: 17.8724535
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -8.0043287, 6.7788167, -8.7491026, 7.3367820, -15.3411102, 15.5279188
1: -31.4100113, 26.4784603, -34.2429581, 28.5470028, -59.9570122, 60.7214012
2: -15.2054825, 23.6200542, -16.6778851, 25.7258816, -40.9313622, 40.2979393
3: -27.4113541, 23.9302959, -29.8492012, 25.8456669, -53.2570076, 53.7794952
4: -20.1638641, 24.4888763, -22.0224686, 26.6615562, -46.8254204, 46.5113449

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8131810, upper bound: 17.8760501
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8117087, upper bound: 17.8225719
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -8.4286814, 7.0882740, -8.2565022, 6.9908934, -15.4195738, 15.3447762
1: -33.0687981, 27.5342007, -32.4270477, 27.3520432, -60.4208412, 59.9612465
2: -16.0192089, 24.8453274, -15.6635942, 24.2992191, -40.3184242, 40.5089149
3: -28.8889580, 24.9061794, -28.2990189, 24.6859169, -53.5748749, 53.2052002
4: -21.2380276, 25.6587963, -20.8182182, 25.2143879, -46.4524155, 46.4770126

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8690294, upper bound: 17.8065985
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8690294, upper bound: 17.8065985
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.7491026, 7.3367820, -8.0043287, 6.7788167, -15.5279188, 15.3411102
1: -34.2429581, 28.5470028, -31.4100113, 26.4784603, -60.7214012, 59.9570122
2: -16.6778851, 25.7258816, -15.2054825, 23.6200542, -40.2979393, 40.9313622
3: -29.8492012, 25.8456669, -27.4113541, 23.9302959, -53.7794952, 53.2570114
4: -22.0224686, 26.6615562, -20.1638641, 24.4888763, -46.5113449, 46.8254204

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8760501, upper bound: 17.8131810
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8225719, upper bound: 17.8117087
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.7259398, 7.3241320, -8.7259398, 7.3241320, -16.0500717, 16.0500717
1: -34.2524223, 28.4322987, -34.2524223, 28.4322987, -62.6847115, 62.6847153
2: -16.5749817, 25.7034645, -16.5749817, 25.7034645, -42.2784462, 42.2784462
3: -29.9225197, 25.7123299, -29.9225197, 25.7123299, -55.6348457, 55.6348495
4: -21.9909878, 26.5092525, -21.9909878, 26.5092525, -48.5002403, 48.5002403

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8710951, upper bound: 17.8951425
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9022690, upper bound: 17.9022873
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8.7259398, 7.3241320, -8.8429289, 7.4126472, -16.1385860, 16.1670609
1: -34.2524223, 28.4322987, -34.6158562, 28.8365402, -63.0889549, 63.0481377
2: -16.5749817, 25.7034645, -16.8528938, 25.9942722, -42.5692520, 42.5563583
3: -29.9225197, 25.7123299, -30.1786442, 26.1047440, -56.0272598, 55.8909760
4: -21.9909878, 26.5092525, -22.2614460, 26.9329090, -48.9238968, 48.7706985

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8710951, upper bound: 17.8955357
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9022690, upper bound: 17.9025302
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.6820068, 7.2847705, -8.7124033, 7.2890673, -15.9710732, 15.9971685
1: -33.9787865, 28.3620148, -34.1786575, 28.2369480, -62.2157364, 62.5406647
2: -16.5510216, 25.5311642, -16.5537319, 25.6913643, -42.2423859, 42.0848846
3: -29.6153545, 25.6866970, -29.8556614, 25.5650139, -55.1803665, 55.5423584
4: -21.8506336, 26.4872246, -21.9379444, 26.4124565, -48.2630920, 48.4251633

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8951269, upper bound: 17.8710772
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8451536, upper bound: 17.8710772
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.7491026, 7.3367820, -8.9976139, 7.5227461, -16.2718468, 16.3343964
1: -34.2429581, 28.5470028, -35.2908859, 29.0765972, -63.3195534, 63.8378792
2: -16.6778851, 25.7258816, -17.0936813, 26.5107250, -43.1886101, 42.8195572
3: -29.8492012, 25.8456669, -30.8474751, 26.2947235, -56.1439209, 56.6931305
4: -22.0224686, 26.6615562, -22.6719246, 27.2264061, -49.2488708, 49.3334808

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8251766, upper bound: 17.8782827
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8235393, upper bound: 17.8234411
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -9.3093596, 7.7925153, -10.6922922, 8.8847475, -18.1941071, 18.4848080
1: -36.5621719, 30.2360554, -42.1563683, 34.5440063, -71.1061707, 72.3924179
2: -17.6497402, 27.3083897, -20.2603512, 31.0164013, -48.6661415, 47.5687408
3: -31.9782372, 27.2964821, -36.8964043, 31.1369095, -63.1151390, 64.1928711
4: -23.5006390, 28.1157818, -27.0730400, 31.9237232, -55.4243622, 55.1888199

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8877782, upper bound: 17.8276619
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8885337, upper bound: 17.8278397
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -8.6387167, 7.2547569, -11.0125475, 9.1496983, -17.7884121, 18.2673035
1: -33.9051285, 28.1681213, -43.3859024, 35.5455666, -69.4506989, 71.5540161
2: -16.4119587, 25.4513779, -20.8706188, 31.9972553, -48.4092102, 46.3219910
3: -29.6192932, 25.4750977, -37.9809380, 32.0357628, -61.6550560, 63.4560356
4: -21.7700729, 26.2575111, -27.8806610, 32.8811302, -54.6512032, 54.1381721

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9021134, upper bound: 17.9018357
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9021070, upper bound: 17.9027771
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -8.7491026, 7.3367820, -11.0125475, 9.1496983, -17.8987999, 18.3493290
1: -34.2429581, 28.5470028, -43.3859024, 35.5455666, -69.7885284, 71.9329071
2: -16.6778851, 25.7258816, -20.8706188, 31.9972553, -48.6751404, 46.5964890
3: -29.8492012, 25.8456669, -37.9809380, 32.0357628, -61.8849640, 63.8266068
4: -22.0224686, 26.6615562, -27.8806610, 32.8811302, -54.9035988, 54.5422173

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8985306, upper bound: 17.8922036
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9012136, upper bound: 17.9018467
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -9.0677299, 7.5930586, -10.7818613, 8.9667845, -18.0345135, 18.3749199
1: -35.5942841, 29.4218216, -42.3870850, 34.8741264, -70.4683990, 71.8088913
2: -17.1829662, 26.6345806, -20.6029797, 31.2613640, -48.4443283, 47.2375603
3: -31.1298256, 26.5819263, -37.0336800, 31.4244041, -62.5542183, 63.6156082
4: -22.8746319, 27.4372578, -27.2729607, 32.2712440, -55.1458740, 54.7102203

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8581190, upper bound: 17.8195655
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8581190, upper bound: 17.8202360
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -9.1508923, 7.6463246, -10.8966217, 9.0588961, -18.2097874, 18.5429459
1: -35.8999138, 29.5508499, -42.8079529, 35.2518044, -71.1517181, 72.3587799
2: -17.3765354, 26.9582958, -20.7888222, 31.6389961, -49.0155296, 47.7471161
3: -31.3756866, 26.7248745, -37.3948975, 31.7701244, -63.1458130, 64.1197662
4: -23.0579720, 27.6747437, -27.5501366, 32.6313591, -55.6893120, 55.2248726

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8945982, upper bound: 17.8956395
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8945399, upper bound: 17.8923491
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -8.7733316, 7.3482757, -10.8914871, 9.0498915, -17.8232193, 18.2397614
1: -34.4277382, 28.4678326, -42.8265076, 35.3467216, -69.7744598, 71.2943420
2: -16.6635876, 25.8361034, -20.8265228, 31.4306297, -48.0942154, 46.6626282
3: -30.0942669, 25.7416077, -37.4119644, 31.8415794, -61.9358444, 63.1535721
4: -22.1205444, 26.5959568, -27.5741386, 32.5744057, -54.6949463, 54.1700974

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9015387, upper bound: 17.8992354
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8946062, upper bound: 17.8969159
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## BFS NS instance: NS_A2_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -10.3442917, 8.6088085, -8.5290003, 7.2018614, -17.5461502, 17.1378098
1: -40.7731094, 33.4841576, -33.4197655, 28.0548611, -68.8279724, 66.9039230
2: -19.6090260, 30.0554676, -16.1730881, 25.0968170, -44.7058411, 46.2285538
3: -35.6872597, 30.1871662, -29.2164707, 25.3409405, -61.0281868, 59.4036369
4: -26.1888733, 30.9412098, -21.5444012, 26.0029602, -52.1918335, 52.4856110

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8290774, upper bound: 17.8869759
time: 0.62 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8300787, upper bound: 17.8908084
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -10.6290026, 8.8467598, -8.6042719, 7.2580996, -17.8871021, 17.4510307
1: -41.8637962, 34.3797073, -33.7158508, 28.2550392, -70.1188278, 68.0955582
2: -20.1537266, 30.9426556, -16.3124905, 25.3192768, -45.4730034, 47.2551460
3: -36.6546097, 30.9903812, -29.4759331, 25.5243053, -62.1789055, 60.4663162
4: -26.9079609, 31.8034859, -21.7340355, 26.2061520, -53.1141090, 53.5375214

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8368886, upper bound: 17.8915220
time: 0.55 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8371357, upper bound: 17.8914694
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -10.4089355, 8.6729393, -8.3742018, 7.0727692, -17.4817047, 17.0471363
1: -40.9015160, 33.7489128, -32.8135681, 27.5301208, -68.4316406, 66.5624847
2: -19.8891602, 30.2257500, -15.8607149, 24.6661682, -44.5553284, 46.0864639
3: -35.7445335, 30.4035931, -28.6805801, 24.8890495, -60.6335831, 59.0841675
4: -26.3336868, 31.2139244, -21.1432514, 25.5566101, -51.8902969, 52.3571777

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_B1

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8229478, upper bound: 17.8509577
time: 0.56 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_B2

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8229478, upper bound: 17.8759771
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -10.7679653, 8.9576902, -8.4503078, 7.1296029, -17.8975677, 17.4079971
1: -42.2981796, 34.8252563, -33.1117744, 27.7380943, -70.0362625, 67.9370270
2: -20.5697651, 31.3123035, -16.0046654, 24.8923969, -45.4621544, 47.3169670
3: -36.9733925, 31.3738003, -28.9427471, 25.0726891, -62.0460701, 60.3165436
4: -27.2401371, 32.2543488, -21.3351192, 25.7619743, -53.0021133, 53.5894623

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8275331, upper bound: 17.8715419
time: 0.62 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B2

### Relational analysis result of NS_A2_B1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8275331, upper bound: 17.8759212
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -11.5897417, 9.5901861, -10.0055847, 8.3133583, -19.9030952, 19.5957661
1: -45.6891136, 37.2044907, -39.3172951, 32.1039619, -77.7930603, 76.5217743
2: -21.9716015, 33.6909523, -18.9267254, 29.2919159, -51.2635117, 52.6176758
3: -39.9875488, 33.5090294, -34.4373779, 28.9725933, -68.9601288, 67.9463959
4: -29.3275127, 34.4562263, -25.2742176, 29.9754181, -59.3029289, 59.7304459

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B2_A1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8217362, upper bound: 17.8903342
time: 0.73 seconds

## Relational analysis of NS_A2_B1_B2_A1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8248460, upper bound: 17.8927504
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -11.6067705, 9.5760765, -9.6040764, 7.9948897, -19.6016598, 19.1801491
1: -45.8330765, 37.3180084, -37.7436218, 30.9426270, -76.7757034, 75.0615997
2: -22.0009747, 33.4624367, -18.1618061, 28.0971928, -50.0981674, 51.6242447
3: -40.0910301, 33.5928383, -33.0566368, 27.9259930, -68.0170212, 66.6494751
4: -29.3991089, 34.3635101, -24.2678452, 28.8253899, -58.2244987, 58.6313515

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B2_A1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8193717, upper bound: 17.8883371
time: 0.72 seconds

## Relational analysis of NS_A2_B1_B2_A1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8173890, upper bound: 17.8921312
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -11.6426172, 9.6460276, -9.0905409, 7.5848660, -19.2274837, 18.7365685
1: -45.8624268, 37.4154243, -35.6922722, 29.3338051, -75.1962280, 73.1076965
2: -22.0674458, 33.8859177, -17.2172832, 26.7283077, -48.7957535, 51.1031990
3: -40.1497498, 33.7050743, -31.2190495, 26.5276299, -66.6773682, 64.9241257
4: -29.4598656, 34.6700478, -22.9221363, 27.4379005, -56.8977661, 57.5921707

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B2_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9023810, upper bound: 17.9025121
time: 0.73 seconds

## Relational analysis of NS_A2_B1_B2_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9023810, upper bound: 17.9025124
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -11.7197447, 9.6839170, -8.7143087, 7.2898293, -19.0095749, 18.3982220
1: -46.2424126, 37.7118607, -34.2324448, 28.2514343, -74.4938431, 71.9442902
2: -22.2218266, 33.8865738, -16.5084362, 25.6122856, -47.8341141, 50.3950119
3: -40.4600754, 33.9653549, -29.9461536, 25.5520859, -66.0121613, 63.9115067
4: -29.6802807, 34.7799644, -21.9877281, 26.3638554, -56.0441360, 56.7676926

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B2_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9026530, upper bound: 17.9025024
time: 0.91 seconds

## Relational analysis of NS_A2_B1_B2_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9026530, upper bound: 17.9025025
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -11.5792618, 9.5945406, -9.1865749, 7.6601124, -19.2393723, 18.7811165
1: -45.6040039, 37.1956711, -35.9828339, 29.6758785, -75.2798615, 73.1785049
2: -21.9604893, 33.7497597, -17.4674873, 26.9974499, -48.9579353, 51.2172470
3: -39.9197693, 33.5158997, -31.4295559, 26.8578243, -66.7775803, 64.9454575
4: -29.2859554, 34.5070229, -23.1565437, 27.7994213, -57.0853767, 57.6635590

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B2_A2_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8994042, upper bound: 17.9025618
time: 0.67 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8994042, upper bound: 17.9026409
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -11.6577101, 9.6350307, -8.8219395, 7.3772678, -19.0349770, 18.4569702
1: -45.9853134, 37.5073853, -34.5591736, 28.6343136, -74.6196060, 72.0665512
2: -22.1152248, 33.7551384, -16.8189983, 25.9434700, -48.0586891, 50.5741348
3: -40.2310944, 33.7876053, -30.1814690, 25.9268990, -66.1579819, 63.9690590
4: -29.5114021, 34.6248665, -22.2425652, 26.7964458, -56.3078461, 56.8674240

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B2_A2_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9005503, upper bound: 17.9025772
time: 0.61 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8923164, upper bound: 17.8970851
time: 0.67 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8922317, upper bound: 17.8922785
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -10.4891205, 8.7764101, -10.4891205, 8.7764101, -19.2655296, 19.2655296
1: -41.2323914, 34.2129707, -41.2323914, 34.2129707, -75.4453583, 75.4453583
2: -19.9594765, 30.6336441, -19.9594765, 30.6336441, -50.5931206, 50.5931206
3: -36.0731812, 30.8062382, -36.0731812, 30.8062382, -66.8794174, 66.8794174
4: -26.5507507, 31.5599766, -26.5507507, 31.5599766, -58.1107254, 58.1107254

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_B1_A1_A1

### Relational analysis result of NS_A2_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8917056, upper bound: 17.8919319
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B1_B1_A1_A2

### Relational analysis result of NS_A2_B2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8919376, upper bound: 17.8919376
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -10.8613405, 9.0572186, -10.4891205, 8.7764101, -19.6377506, 19.5463390
1: -42.7690125, 35.4195366, -41.2323914, 34.2129707, -76.9819794, 76.6519318
2: -20.7385616, 31.4904613, -19.9594765, 30.6336441, -51.3722000, 51.4499359
3: -37.4092827, 31.8676853, -36.0731812, 30.8062382, -68.2155228, 67.9408569
4: -27.5132542, 32.5443573, -26.5507507, 31.5599766, -59.0732307, 59.0951042

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_B1_A2_A1

### Relational analysis result of NS_A2_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8917056, upper bound: 17.8919612
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_B1_A2_A2

### Relational analysis result of NS_A2_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8919376, upper bound: 17.8919410
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -10.4891205, 8.7764101, -10.8613405, 9.0572186, -19.5463390, 19.6377487
1: -41.2323914, 34.2129707, -42.7690125, 35.4195366, -76.6519318, 76.9819794
2: -19.9594765, 30.6336441, -20.7385616, 31.4904613, -51.4499359, 51.3722000
3: -36.0731812, 30.8062382, -37.4092827, 31.8676853, -67.9408569, 68.2155228
4: -26.5507507, 31.5599766, -27.5132542, 32.5443573, -59.0951042, 59.0732307

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8919319, upper bound: 17.8936617
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8919376, upper bound: 17.8921897
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -10.8613405, 9.0572186, -10.8613405, 9.0572186, -19.9185600, 19.9185600
1: -42.7690125, 35.4195366, -42.7690125, 35.4195366, -78.1885529, 78.1885529
2: -20.7385616, 31.4904613, -20.7385616, 31.4904613, -52.2290230, 52.2290230
3: -37.4092827, 31.8676853, -37.4092827, 31.8676853, -69.2769623, 69.2769547
4: -27.5132542, 32.5443573, -27.5132542, 32.5443573, -60.0576096, 60.0576096

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8917056, upper bound: 17.8922134
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8919376, upper bound: 17.8921931
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -10.4891205, 8.7764101, -11.9412880, 9.8914385, -20.3805580, 20.7176971
1: -41.2323914, 34.2129707, -47.0488510, 38.3466263, -79.5790024, 81.2618256
2: -19.9594765, 30.6336441, -22.6441879, 34.8102417, -54.7697182, 53.2778282
3: -36.0731812, 30.8062382, -41.1872787, 34.5389404, -70.6121140, 71.9935150
4: -26.5507507, 31.5599766, -30.2141342, 35.5589371, -62.1096840, 61.7741089

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8920291, upper bound: 17.8964392
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8919817, upper bound: 17.8926640
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -10.8613405, 9.0572186, -11.9412880, 9.8914385, -20.7527790, 20.9985065
1: -42.7690125, 35.4195366, -47.0488510, 38.3466263, -81.1156387, 82.4683838
2: -20.7385616, 31.4904613, -22.6441879, 34.8102417, -55.5488052, 54.1346512
3: -37.4092827, 31.8676853, -41.1872787, 34.5389404, -71.9482117, 73.0549622
4: -27.5132542, 32.5443573, -30.2141342, 35.5589371, -63.0721893, 62.7584839

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8994313, upper bound: 17.9024035
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8994373, upper bound: 17.9023337
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -10.4891205, 8.7764101, -12.0305853, 9.9370108, -20.4261322, 20.8069954
1: -41.2323914, 34.2129707, -47.4764481, 38.6831589, -79.9155502, 81.6894226
2: -19.9594765, 30.6336441, -22.8127365, 34.8363533, -54.7958298, 53.4463730
3: -36.0731812, 30.8062382, -41.5374870, 34.8268051, -70.8999863, 72.3437271
4: -26.5507507, 31.5599766, -30.4614754, 35.6952209, -62.2459602, 62.0214500

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8920270, upper bound: 17.8975311
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8919451, upper bound: 17.8922425
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -10.8613405, 9.0572186, -12.0305853, 9.9370108, -20.7983494, 21.0878029
1: -42.7690125, 35.4195366, -47.4764481, 38.6831589, -81.4521713, 82.8959808
2: -20.7385616, 31.4904613, -22.8127365, 34.8363533, -55.5749130, 54.3031960
3: -37.4092827, 31.8676853, -41.5374870, 34.8268051, -72.2360840, 73.4051743
4: -27.5132542, 32.5443573, -30.4614754, 35.6952209, -63.2084694, 63.0058327

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8994160, upper bound: 17.9026021
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8994230, upper bound: 17.9026024
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -11.9412880, 9.8914385, -10.4891205, 8.7764101, -20.7176971, 20.3805580
1: -47.0488510, 38.3466263, -41.2323914, 34.2129707, -81.2618256, 79.5790024
2: -22.6441879, 34.8102417, -19.9594765, 30.6336441, -53.2778244, 54.7697182
3: -41.1872787, 34.5389404, -36.0731812, 30.8062382, -71.9935150, 70.6121063
4: -30.2141342, 35.5589371, -26.5507507, 31.5599766, -61.7741089, 62.1096802

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8917056, upper bound: 17.8920299
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8926640, upper bound: 17.8919817
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -11.9412880, 9.8914385, -10.8613405, 9.0572186, -20.9985065, 20.7527790
1: -47.0488510, 38.3466263, -42.7690125, 35.4195366, -82.4683838, 81.1156311
2: -22.6441879, 34.8102417, -20.7385616, 31.4904613, -54.1346474, 55.5488052
3: -41.1872787, 34.5389404, -37.4092827, 31.8676853, -73.0549622, 71.9482117
4: -30.2141342, 35.5589371, -27.5132542, 32.5443573, -62.7584915, 63.0721893

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9028534, upper bound: 17.9005785
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9023377, upper bound: 17.9005789
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -12.0305853, 9.9370108, -10.4891205, 8.7764101, -20.8069954, 20.4261322
1: -47.4764481, 38.6831589, -41.2323914, 34.2129707, -81.6894226, 79.9155502
2: -22.8127365, 34.8363533, -19.9594765, 30.6336441, -53.4463730, 54.7958298
3: -41.5374870, 34.8268051, -36.0731812, 30.8062382, -72.3437271, 70.8999786
4: -30.4614754, 35.6952209, -26.5507507, 31.5599766, -62.0214462, 62.2459564

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9002800, upper bound: 17.8920320
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8922425, upper bound: 17.8919451
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -12.0305853, 9.9370108, -10.8613405, 9.0572186, -21.0878029, 20.7983494
1: -47.4764481, 38.6831589, -42.7690125, 35.4195366, -82.8959808, 81.4521713
2: -22.8127365, 34.8363533, -20.7385616, 31.4904613, -54.3031960, 55.5749130
3: -41.5374870, 34.8268051, -37.4092827, 31.8676853, -73.4051743, 72.2360840
4: -30.4614754, 35.6952209, -27.5132542, 32.5443573, -63.0058327, 63.2084656

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9032066, upper bound: 17.9005646
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9026096, upper bound: 17.9005646
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -11.9412880, 9.8914385, -11.9412880, 9.8914385, -21.8327255, 21.8327255
1: -47.0488510, 38.3466263, -47.0488510, 38.3466263, -85.3954773, 85.3954773
2: -22.6441879, 34.8102417, -22.6441879, 34.8102417, -57.4544296, 57.4544296
3: -41.1872787, 34.5389404, -41.1872787, 34.5389404, -75.7262115, 75.7262115
4: -30.2141342, 35.5589371, -30.2141342, 35.5589371, -65.7730637, 65.7730637

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8969929, upper bound: 17.9025889
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9029806, upper bound: 17.9029666
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -11.9412880, 9.8914385, -12.0305853, 9.9370108, -21.8782978, 21.9220238
1: -47.0488510, 38.3466263, -47.4764481, 38.6831589, -85.7320099, 85.8230591
2: -22.6441879, 34.8102417, -22.8127365, 34.8363533, -57.4805412, 57.6229744
3: -41.1872787, 34.5389404, -41.5374870, 34.8268051, -76.0140839, 76.0764313
4: -30.2141342, 35.5589371, -30.4614754, 35.6952209, -65.9093475, 66.0203857

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8969929, upper bound: 17.9025982
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9029806, upper bound: 17.9031598
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -12.0305853, 9.9370108, -11.9412880, 9.8914385, -21.9220238, 21.8782978
1: -47.4764481, 38.6831589, -47.0488510, 38.3466263, -85.8230667, 85.7320099
2: -22.8127365, 34.8363533, -22.6441879, 34.8102417, -57.6229744, 57.4805374
3: -41.5374870, 34.8268051, -41.1872787, 34.5389404, -76.0764313, 76.0140839
4: -30.4614754, 35.6952209, -30.2141342, 35.5589371, -66.0203934, 65.9093475

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8978482, upper bound: 17.8954120
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9031860, upper bound: 17.9029635
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -12.0305853, 9.9370108, -12.0305853, 9.9370108, -21.9675961, 21.9675961
1: -47.4764481, 38.6831589, -47.4764481, 38.6831589, -86.1596069, 86.1596069
2: -22.8127365, 34.8363533, -22.8127365, 34.8363533, -57.6490860, 57.6490860
3: -41.5374870, 34.8268051, -41.5374870, 34.8268051, -76.3642883, 76.3642883
4: -30.4614754, 35.6952209, -30.4614754, 35.6952209, -66.1566772, 66.1566772

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8955652, upper bound: 17.9027550
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9031860, upper bound: 17.9031541
time: 0.70 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.13 seconds
NS_A1_B1_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8065985, upper bound: 17.8690294
NS_A1_B1_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8065985, upper bound: 17.8724535
NS_A1_B1_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8131810, upper bound: 17.8760501
NS_A1_B1_A1_B2_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8117087, upper bound: 17.8225719
NS_A1_B1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8690294, upper bound: 17.8065985
NS_A1_B1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8690294, upper bound: 17.8065985
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8760501, upper bound: 17.8131810
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8225719, upper bound: 17.8117087
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8710951, upper bound: 17.8951425
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.9022690, upper bound: 17.9022873
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8710951, upper bound: 17.8955357
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.9022690, upper bound: 17.9025302
NS_A1_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8951269, upper bound: 17.8710772
NS_A1_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8451536, upper bound: 17.8710772
NS_A1_B1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8251766, upper bound: 17.8782827
NS_A1_B1_A2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8235393, upper bound: 17.8234411
NS_A1_B2_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8877782, upper bound: 17.8276619
NS_A1_B2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8885337, upper bound: 17.8278397
NS_A1_B2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.9021134, upper bound: 17.9018357
NS_A1_B2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.9021070, upper bound: 17.9027771
NS_A1_B2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8985306, upper bound: 17.8922036
NS_A1_B2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.9012136, upper bound: 17.9018467
NS_A1_B2_A2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8581190, upper bound: 17.8195655
NS_A1_B2_A2_B2_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8581190, upper bound: 17.8202360
NS_A1_B2_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8945982, upper bound: 17.8956395
NS_A1_B2_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8945399, upper bound: 17.8923491
NS_A2_B1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8290774, upper bound: 17.8869759
NS_A2_B1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8300787, upper bound: 17.8908084
NS_A2_B1_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8368886, upper bound: 17.8915220
NS_A2_B1_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8371357, upper bound: 17.8914694
NS_A2_B1_B1_A2_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8229478, upper bound: 17.8509577
NS_A2_B1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8229478, upper bound: 17.8759771
NS_A2_B1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8275331, upper bound: 17.8715419
NS_A2_B1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8275331, upper bound: 17.8759212
NS_A2_B1_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8217362, upper bound: 17.8903342
NS_A2_B1_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8248460, upper bound: 17.8927504
NS_A2_B1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8193717, upper bound: 17.8883371
NS_A2_B1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8173890, upper bound: 17.8921312
NS_A2_B1_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.9023810, upper bound: 17.9025121
NS_A2_B1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.9023810, upper bound: 17.9025124
NS_A2_B1_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.9026530, upper bound: 17.9025024
NS_A2_B1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.9026530, upper bound: 17.9025025
NS_A2_B1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8994042, upper bound: 17.9025618
NS_A2_B1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8994042, upper bound: 17.9026409
NS_A2_B1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8923164, upper bound: 17.8970851
NS_A2_B1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8922317, upper bound: 17.8922785
NS_A2_B2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8917056, upper bound: 17.8919319
NS_A2_B2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8919376, upper bound: 17.8919376
NS_A2_B2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8917056, upper bound: 17.8919612
NS_A2_B2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8919376, upper bound: 17.8919410
NS_A2_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8919319, upper bound: 17.8936617
NS_A2_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8919376, upper bound: 17.8921897
NS_A2_B2_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8917056, upper bound: 17.8922134
NS_A2_B2_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8919376, upper bound: 17.8921931
NS_A2_B2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8920291, upper bound: 17.8964392
NS_A2_B2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8919817, upper bound: 17.8926640
NS_A2_B2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8994313, upper bound: 17.9024035
NS_A2_B2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8994373, upper bound: 17.9023337
NS_A2_B2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8920270, upper bound: 17.8975311
NS_A2_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8919451, upper bound: 17.8922425
NS_A2_B2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8994160, upper bound: 17.9026021
NS_A2_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8994230, upper bound: 17.9026024
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8917056, upper bound: 17.8920299
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8926640, upper bound: 17.8919817
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.9028534, upper bound: 17.9005785
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.9023377, upper bound: 17.9005789
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.9002800, upper bound: 17.8920320
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8922425, upper bound: 17.8919451
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.9032066, upper bound: 17.9005646
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.9026096, upper bound: 17.9005646
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8969929, upper bound: 17.9025889
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.9029806, upper bound: 17.9029666
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8969929, upper bound: 17.9025982
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.9029806, upper bound: 17.9031598
NS_A2_B2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8978482, upper bound: 17.8954120
NS_A2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.9031860, upper bound: 17.9029635
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.8955652, upper bound: 17.9027550
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 0, lower bound: -17.9031860, upper bound: 17.9031541

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -7.3881474, 6.3283181, -8.4286814, 7.0882740, -14.4764214, 14.7569962
1: -29.0069828, 24.9005146, -33.0687981, 27.5342007, -56.5411835, 57.9693108
2: -14.0104818, 21.8076992, -16.0192089, 24.8453274, -38.8558083, 37.8269081
3: -25.2967243, 22.4820232, -28.8889580, 24.9061794, -50.2028923, 51.3709793
4: -18.6195602, 22.8523388, -21.2380276, 25.6587963, -44.2783585, 44.0903664

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7795453, upper bound: 17.8206934
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7981381, upper bound: 17.8669185
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -7.9579811, 6.7475147, -8.4286814, 7.0882740, -15.0462542, 15.1761961
1: -31.1718979, 26.5536346, -33.0687981, 27.5342007, -58.7061005, 59.6224327
2: -15.2077475, 23.4306622, -16.0192089, 24.8453274, -40.0530739, 39.4498711
3: -27.1192417, 24.0276909, -28.8889580, 24.9061794, -52.0254135, 52.9166489
4: -20.0351963, 24.5631638, -21.2380276, 25.6587963, -45.6939926, 45.8011932

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7795453, upper bound: 17.8206934
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7981381, upper bound: 17.8702605
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.9619884, 6.7458057, -8.7848101, 7.3683968, -15.3303843, 15.5306149
1: -31.2448177, 26.3512478, -34.3980827, 28.7207966, -59.9656143, 60.7493286
2: -15.1263962, 23.5010567, -16.7129002, 25.7305336, -40.8569221, 40.2139587
3: -27.2682362, 23.8147335, -30.0003548, 25.9900322, -53.2582703, 53.8150864
4: -20.0592480, 24.3686218, -22.1344376, 26.7464256, -46.8056717, 46.5030594

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8012251, upper bound: 17.8682764
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8012251, upper bound: 17.8760501
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -8.4286814, 7.0882740, -7.3881474, 6.3283181, -14.7569962, 14.4764204
1: -33.0687981, 27.5342007, -29.0069828, 24.9005146, -57.9693069, 56.5411797
2: -16.0192089, 24.8453274, -14.0104818, 21.8076992, -37.8269081, 38.8558083
3: -28.8889580, 24.9061794, -25.2967243, 22.4820232, -51.3709793, 50.2028923
4: -21.2380276, 25.6587963, -18.6195602, 22.8523388, -44.0903664, 44.2783585

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8689255, upper bound: 17.8012253
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8294955, upper bound: 17.8000869
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -8.4286814, 7.0882740, -7.9579811, 6.7475147, -15.1761951, 15.0462551
1: -33.0687981, 27.5342007, -31.1718979, 26.5536346, -59.6224289, 58.7061005
2: -16.0192089, 24.8453274, -15.2077475, 23.4306622, -39.4498711, 40.0530739
3: -28.8889580, 24.9061794, -27.1192417, 24.0276909, -52.9166489, 52.0254135
4: -21.2380276, 25.6587963, -20.0351963, 24.5631638, -45.8011932, 45.6939888

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8689255, upper bound: 17.8129428
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8294955, upper bound: 17.8117454
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.7848101, 7.3683968, -7.9619884, 6.7458057, -15.5306149, 15.3303843
1: -34.3980827, 28.7207966, -31.2448177, 26.3512478, -60.7493286, 59.9656143
2: -16.7129002, 25.7305336, -15.1263962, 23.5010567, -40.2139587, 40.8569221
3: -30.0003548, 25.9900322, -27.2682362, 23.8147335, -53.8150864, 53.2582703
4: -22.1344376, 26.7464256, -20.0592480, 24.3686218, -46.5030594, 46.8056717

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8682764, upper bound: 17.8012251
time: 2.69 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8682764, upper bound: 17.8131755
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.1457329, 6.8631196, -8.5366135, 7.1761785, -15.3219109, 15.3997326
1: -31.9718437, 26.7109318, -33.5057373, 27.8833523, -59.8551941, 60.2166672
2: -15.4846220, 24.0333576, -16.2163982, 25.1526146, -40.6372299, 40.2497559
3: -27.9176750, 24.1798592, -29.2672081, 25.2216473, -53.1393204, 53.4470673
4: -20.5096245, 24.8463383, -21.5093784, 25.9719162, -46.4815407, 46.3557129

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8281669, upper bound: 17.8822148
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8273971, upper bound: 17.8325640
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.4286814, 7.0882740, -8.6387167, 7.2547569, -15.6834383, 15.7269888
1: -33.0687981, 27.5342007, -33.9051285, 28.1681213, -61.2369194, 61.4393234
2: -16.0192089, 24.8453274, -16.4119587, 25.4513779, -41.4705811, 41.2572861
3: -28.8889580, 24.9061794, -29.6192932, 25.4750977, -54.3640556, 54.5254707
4: -21.2380276, 25.6587963, -21.7700729, 26.2575111, -47.4955368, 47.4288712

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8333906, upper bound: 17.8831930
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8325940, upper bound: 17.8325987
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.1457329, 6.8631196, -8.6820068, 7.2847705, -15.4305000, 15.5451260
1: -31.9718437, 26.7109318, -33.9787865, 28.3620148, -60.3338547, 60.6897125
2: -15.4846220, 24.0333576, -16.5510216, 25.5311642, -41.0157852, 40.5843811
3: -27.9176750, 24.1798592, -29.6153545, 25.6866970, -53.6043701, 53.7952118
4: -20.5096245, 24.8463383, -21.8506336, 26.4872246, -46.9968452, 46.6969681

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8704047, upper bound: 17.8704047
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8704047, upper bound: 17.8955358
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.4286814, 7.0882740, -8.7491026, 7.3367820, -15.7654629, 15.8373747
1: -33.0687981, 27.5342007, -34.2429581, 28.5470028, -61.6157990, 61.7771606
2: -16.0192089, 24.8453274, -16.6778851, 25.7258816, -41.7450829, 41.5232124
3: -28.8889580, 24.9061794, -29.8492012, 25.8456669, -54.7346153, 54.7553749
4: -21.2380276, 25.6587963, -22.0224686, 26.6615562, -47.8995819, 47.6812630

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8844420, upper bound: 17.8251063
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8307824, upper bound: 17.8235778
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -8.6820068, 7.2847705, -8.1308079, 6.8494964, -15.5315037, 15.4155731
1: -33.9787865, 28.3620148, -31.9131050, 26.6543941, -60.6331787, 60.2751198
2: -16.5510216, 25.5311642, -15.4563532, 23.9873619, -40.5383835, 40.9875183
3: -29.6153545, 25.6866970, -27.8665028, 24.1282692, -53.7436218, 53.5531998
4: -21.8506336, 26.4872246, -20.4719582, 24.7973156, -46.6479454, 46.9591751

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7542056, upper bound: 17.8704774
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7542056, upper bound: 17.8710772
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -8.6820068, 7.2847705, -8.2468567, 6.9428830, -15.6248894, 15.5316238
1: -33.9787865, 28.3620148, -32.2619476, 27.0497894, -61.0285759, 60.6239624
2: -16.5510216, 25.5311642, -15.7451563, 24.3248615, -40.8758850, 41.2763176
3: -29.6153545, 25.6866970, -28.1239586, 24.5155373, -54.1308861, 53.8106537
4: -21.8506336, 26.4872246, -20.7459602, 25.2636318, -47.1142616, 47.2331772

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8783849, upper bound: 17.8405326
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7721060, upper bound: 17.8710574
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -8.6930199, 7.2934504, -8.9719639, 7.5156455, -16.2086658, 16.2654114
1: -34.0215187, 28.3832436, -35.2069550, 29.1042633, -63.1257820, 63.5901985
2: -16.5720234, 25.5702381, -17.0103416, 26.3666401, -42.9386635, 42.5805740
3: -29.6632919, 25.6964207, -30.7969990, 26.2928238, -55.9561157, 56.4934120
4: -21.8825531, 26.5025368, -22.6332512, 27.1526451, -49.0351982, 49.1357880

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8244769, upper bound: 17.8644064
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8244769, upper bound: 17.8644064
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -8.0999365, 6.9004989, -10.2592478, 8.5430422, -16.6429787, 17.1597462
1: -31.7121010, 27.0063972, -40.4384193, 33.2423935, -64.9544983, 67.4448090
2: -15.4276829, 23.9066296, -19.4494686, 29.8008842, -45.2285690, 43.3560944
3: -27.6898384, 24.3783283, -35.3922310, 29.9675140, -57.6573486, 59.7705536
4: -20.4504700, 24.9375114, -25.9750099, 30.7049122, -51.1553764, 50.9125175

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8826806, upper bound: 17.8128209
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8868565, upper bound: 17.8201060
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -9.0485229, 7.5904779, -10.6629105, 8.8616323, -17.9101562, 18.2533875
1: -35.5307465, 29.4758110, -42.0405045, 34.4568634, -69.9876022, 71.5163116
2: -17.1565228, 26.5703354, -20.2053986, 30.9328461, -48.0893593, 46.7757225
3: -31.0749855, 26.6097832, -36.7947655, 31.0583763, -62.1333618, 63.4045486
4: -22.8387375, 27.3847523, -26.9987469, 31.8412056, -54.6799431, 54.3834991

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_A2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8885337, upper bound: 17.8278397
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_A2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8885337, upper bound: 17.8278397
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.6038523, 7.2279849, -10.7290430, 8.9289742, -17.5328236, 17.9570274
1: -33.7661667, 28.0696430, -42.2597160, 34.7278442, -68.4940033, 70.3293610
2: -16.3441124, 25.3534794, -20.3170872, 31.1876926, -47.5318069, 45.6705666
3: -29.4953194, 25.3870831, -36.9729843, 31.3034077, -60.7987213, 62.3600693
4: -21.6802177, 26.1622810, -27.1515484, 32.0998039, -53.7800217, 53.3138275

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8961107, upper bound: 17.8917387
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9011031, upper bound: 17.9004706
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8.2418718, 6.9450493, -10.7743778, 8.9402275, -17.1820984, 17.7194233
1: -32.3551331, 27.0288525, -42.5180473, 34.9246597, -67.2797928, 69.5468750
2: -15.6624012, 24.2750797, -20.4132004, 31.0678482, -46.7302475, 44.6882706
3: -28.2716637, 24.4460793, -37.1966934, 31.4775982, -59.7492599, 61.6427689
4: -20.7800121, 25.1246681, -27.3106899, 32.1175194, -52.8975258, 52.4353561

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8844458, upper bound: 17.8960422
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9010968, upper bound: 17.9018011
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.5017662, 7.1564045, -9.7671995, 8.2320366, -16.7338028, 16.9236031
1: -33.2469406, 27.8931980, -38.3507271, 32.1813812, -65.4283142, 66.2439194
2: -16.2285538, 25.0617027, -18.5813141, 28.5539932, -44.7825317, 43.6430130
3: -28.9581146, 25.2624264, -33.4928932, 28.9895935, -57.9477081, 58.7553177
4: -21.3875542, 26.0298977, -24.7099285, 29.5801239, -50.9676743, 50.7398186

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8855107, upper bound: 17.8143454
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8913069, upper bound: 17.8889207
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8987242, upper bound: 17.8919886
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.7419729, 7.3314428, -10.5975847, 8.8270779, -17.5690498, 17.9290237
1: -34.2142372, 28.5274906, -41.7469864, 34.3477859, -68.5620270, 70.2744675
2: -16.6649494, 25.7061634, -20.0972137, 30.7844753, -47.4494209, 45.8033714
3: -29.8242245, 25.8279343, -36.5411682, 30.9469776, -60.7712021, 62.3691025
4: -22.0046234, 26.6425285, -26.8461571, 31.7144718, -53.7190933, 53.4886856

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9013737, upper bound: 17.9004845
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9013674, upper bound: 17.9018150
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -9.0546885, 7.5751390, -11.2852621, 9.4397078, -18.4943943, 18.8604012
1: -35.5190582, 29.2934704, -44.3309822, 36.7462883, -72.2653275, 73.6244507
2: -17.1928043, 26.6865292, -21.5075645, 32.9931717, -50.1859741, 48.1940918
3: -31.0409050, 26.4904118, -38.7219887, 33.1008835, -64.1417847, 65.2123871
4: -22.8140469, 27.4161282, -28.5484314, 34.0081596, -56.8222046, 55.9645615

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8929838, upper bound: 17.8947764
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8909531, upper bound: 17.8946946
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -9.1508923, 7.6463246, -10.6373119, 8.8624458, -18.0133381, 18.2836361
1: -35.8999138, 29.5508499, -41.7818184, 34.5091896, -70.4091034, 71.3326569
2: -17.3765354, 26.9582958, -20.2854614, 30.9280529, -48.3045883, 47.2437592
3: -31.3756866, 26.7248745, -36.5046425, 31.1046486, -62.4803352, 63.2295151
4: -23.0579720, 27.6747437, -26.8791561, 31.9264374, -54.9844055, 54.5538940

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8841825, upper bound: 17.8920141
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8841825, upper bound: 17.8923491
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -9.1393709, 7.7294803, -8.2542019, 7.0083714, -16.1477375, 15.9836817
1: -35.8738098, 30.2788334, -32.2918816, 27.3588486, -63.2326584, 62.5707169
2: -17.4019070, 26.7787361, -15.6863480, 24.3614388, -41.7633438, 42.4650803
3: -31.3199558, 27.2880611, -28.2161217, 24.7057667, -56.0257187, 55.5041809
4: -23.1133995, 27.7929554, -20.8414612, 25.3124542, -48.4258537, 48.6344147

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8290774, upper bound: 17.8869759
time: 0.60 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8290774, upper bound: 17.8869759
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -9.9395943, 8.2911854, -8.5233078, 7.1975288, -17.1371231, 16.8144932
1: -39.1652031, 32.3045044, -33.3970032, 28.0388279, -67.2040329, 65.7015076
2: -18.8505039, 28.8668041, -16.1623955, 25.0806618, -43.9311600, 45.0291977
3: -34.2691803, 29.1173592, -29.1963806, 25.3264732, -59.5956421, 58.3137398
4: -25.1700764, 29.7976952, -21.5300159, 25.9874001, -51.1574783, 51.3277130

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8241150, upper bound: 17.8833285
time: 0.65 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8231525, upper bound: 17.8816416
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -10.3528042, 8.6321507, -8.5723572, 7.2336993, -17.5865040, 17.2045040
1: -40.7558212, 33.5831871, -33.5893860, 28.1656151, -68.9214249, 67.1725693
2: -19.6141186, 30.1560192, -16.2509613, 25.2290916, -44.8432083, 46.4069824
3: -35.6703720, 30.2778568, -29.3645878, 25.4438610, -61.1142235, 59.6424446
4: -26.1954632, 31.0440102, -21.6532097, 26.1194649, -52.3149261, 52.6972198

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8366294, upper bound: 17.8912790
time: 0.68 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8366294, upper bound: 17.8914694
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -10.4063044, 8.6499062, -8.2596693, 6.9899411, -17.3962421, 16.9095726
1: -41.0528793, 33.8058167, -32.3626137, 27.2732410, -68.3261032, 66.1684265
2: -19.7202187, 30.0518894, -15.6745644, 24.2999649, -44.0201797, 45.7264557
3: -35.9200287, 30.4745674, -28.2905350, 24.6334457, -60.5534744, 58.7650909
4: -26.3759651, 31.0845852, -20.8650837, 25.2481060, -51.6240654, 51.9496689

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8371358, upper bound: 17.8912790
time: 0.73 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8371357, upper bound: 17.8914694
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -10.2854071, 8.5739183, -7.7094221, 6.5571332, -16.8425388, 16.2833405
1: -40.4143753, 33.3832970, -30.1876507, 25.6375828, -66.0519562, 63.5709381
2: -19.6554871, 29.8462315, -14.6390429, 22.7265415, -42.3820267, 44.4852753
3: -35.3133926, 30.0775185, -26.3686218, 23.1877155, -58.5011024, 56.4461327
4: -26.0212841, 30.8605556, -19.4587898, 23.7191010, -49.7403793, 50.3193436

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_B2_B1

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8236586, upper bound: 17.8602686
time: 0.65 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_B2_B2

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8236586, upper bound: 17.8759771
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -10.4190178, 8.6810608, -7.3584371, 6.2572007, -16.6762180, 16.0394936
1: -40.9160957, 33.7468491, -28.7872028, 24.3153267, -65.2314224, 62.5340500
2: -19.9085522, 30.3463802, -13.9699259, 21.7765484, -41.6851006, 44.3163033
3: -35.7897415, 30.3942413, -25.1686611, 21.9685879, -57.7583237, 55.5628929
4: -26.3553505, 31.2471828, -18.5716057, 22.5971642, -48.9525108, 49.8187866

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8275331, upper bound: 17.8715419
time: 0.69 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8275331, upper bound: 17.8715419
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -10.7679653, 8.9576902, -8.3042440, 7.0174279, -17.7853928, 17.2619286
1: -42.2981796, 34.8252563, -32.5307541, 27.3138123, -69.6119766, 67.3560104
2: -20.5697651, 31.3123035, -15.7313404, 24.4805431, -45.0503082, 47.0436440
3: -36.9733925, 31.3738003, -28.4315834, 24.6919823, -61.6653748, 59.8053818
4: -27.2401371, 32.2543488, -20.9655571, 25.3597584, -52.5998955, 53.2199020

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8275611, upper bound: 17.8759212
time: 0.72 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8275331, upper bound: 17.8759212
time: 1.24 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.13 + 417.99 = 421.12 seconds
